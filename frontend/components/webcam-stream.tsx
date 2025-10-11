"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Video, VideoOff } from "lucide-react"
import { PersonNotification } from "@/components/person-notification"

interface PersonData {
  name: string
  description: string
  relationship: string
}

export default function WebcamStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [personData, setPersonData] = useState<PersonData | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const BACKEND_URL = "http://localhost:8000"

  useEffect(() => {
    startWebcam()
    connectSSE()

    return () => {
      stopWebcam()
      disconnectSSE()
    }
  }, [])

  // Connect to SSE for AI responses
  const connectSSE = () => {
    try {
      const sessionId = Math.random().toString(36).substring(7)
      const es = new EventSource(`${BACKEND_URL}/events?session_id=${sessionId}`)

      console.log('[SSE] Connecting to:', `${BACKEND_URL}/events`)

      es.onopen = () => {
        console.log('[SSE] Connected')
      }

      es.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('[SSE] Received:', message)

          // Handle person data (simplified structure)
          if (message.name && message.description && message.relationship) {
            setPersonData({
              name: message.name,
              description: message.description,
              relationship: message.relationship,
            })
          }
        } catch (err) {
          console.error('[SSE] Error parsing message:', err)
        }
      }

      es.onerror = (error) => {
        console.error('[SSE] Error:', error)
        // SSE will auto-reconnect
      }

      setEventSource(es)
    } catch (err) {
      console.error('[SSE] Connection error:', err)
    }
  }

  // Disconnect SSE
  const disconnectSSE = () => {
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
  }

  // Wait for ICE gathering to complete
  const waitForIceGathering = (pc: RTCPeerConnection): Promise<void> => {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') {
        resolve()
        return
      }

      const checkState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', checkState)
          resolve()
        }
      }

      pc.addEventListener('icegatheringstatechange', checkState)
    })
  }

  // Setup WebRTC connection
  const setupWebRTC = async (stream: MediaStream) => {
    try {
      console.log('[WebRTC] Setting up peer connection')
      const pc = new RTCPeerConnection()
      pcRef.current = pc

      // Add all tracks (video + audio) to peer connection
      stream.getTracks().forEach(track => {
        console.log('[WebRTC] Adding track:', track.kind)
        pc.addTrack(track, stream)
      })

      // Monitor connection state
      pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState)
        setIsConnected(pc.connectionState === 'connected')
      }

      // Create and send offer
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      console.log('[WebRTC] Created offer, waiting for ICE gathering...')

      // Wait for ICE gathering
      await waitForIceGathering(pc)
      console.log('[WebRTC] ICE gathering complete, sending offer to backend')

      // Send offer to backend
      const response = await fetch(`${BACKEND_URL}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription!.sdp,
          type: pc.localDescription!.type
        })
      })

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`)
      }

      const answer = await response.json()
      console.log('[WebRTC] Received answer from backend')
      await pc.setRemoteDescription(answer)
      console.log('[WebRTC] Connection established!')

    } catch (err) {
      console.error('[WebRTC] Setup error:', err)
      setIsConnected(false)
    }
  }

  // Start webcam
  const startWebcam = async () => {
    try {
      console.log('[Webcam] Requesting media access...')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,  // Enable audio for WebRTC
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        console.log('[Webcam] Stream started')
        
        // Setup WebRTC connection after a brief delay to ensure video is ready
        setTimeout(() => setupWebRTC(stream), 1000)
      }
    } catch (err) {
      console.error("[Webcam] Error accessing webcam:", err)
    }
  }

  // Stop webcam
  const stopWebcam = () => {
    console.log('[Webcam] Stopping stream')
    
    // Stop media tracks
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    
    // Close peer connection
    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }
    
    setIsStreaming(false)
    setIsConnected(false)
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Video Background */}
      <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 h-full w-full object-cover" />

      {/* WebRTC Connection Status */}
      <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} ${isConnected ? 'animate-pulse' : ''}`} />
        <span className="text-xs text-white/80">{isConnected ? 'Connected (WebRTC)' : 'Disconnected'}</span>
      </div>

      {/* Person Notification */}
      {personData && (
        <PersonNotification
          name={personData.name}
          description={personData.description}
          relationship={personData.relationship}
          onClose={() => setPersonData(null)}
        />
      )}

      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center px-6 py-4 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex items-center gap-3">
          <Button
            size="icon"
            variant={isStreaming ? "default" : "secondary"}
            onClick={isStreaming ? stopWebcam : startWebcam}
            className="h-12 w-12 rounded-full"
          >
            {isStreaming ? <Video className="h-5 w-5" /> : <VideoOff className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isStreaming ? "Stop Video" : "Start Video"}</span>
        </div>
      </div>
    </div>
  )
}
