"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Mic, MicOff, Video, VideoOff } from "lucide-react"
import { FaceNotification } from "@/components/face-notification"
import { useFaceDetection } from "@/hooks/use-face-detection"
import { RayBanOverlay } from "@/components/rayban-overlay"
import { cn } from "@/lib/utils"
import {
  calculateVideoTransform,
  mapBoundingBoxToOverlay,
  calculateNotificationPosition,
} from "@/lib/coordinate-mapper"

interface PersonData {
  name: string
  description: string
  relationship: string
  person_id?: string
}

type FacePersonMap = Map<string, PersonData>

export default function WebcamStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [facePersonData, setFacePersonData] = useState<FacePersonMap>(new Map())
  const [latestPersonData, setLatestPersonData] = useState<PersonData | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [isRayBanMode, setIsRayBanMode] = useState(false)
  const [isMuted, setIsMuted] = useState(false)

  const INFERENCE_BACKEND_URL = "http://localhost:8002"
  const OFFER_BACKEND_URL = "http://localhost:8000"

  const { detectedFaces, isLoading: isFaceDetectionLoading, error: faceDetectionError } = useFaceDetection(
    videoRef.current,
    {
      enabled: isStreaming && isVideoReady && !isRayBanMode,
      minDetectionConfidence: 0.5,
      targetFps: 20,
      useWorker: true,
    }
  )

  useEffect(() => {
    if (!latestPersonData || detectedFaces.length === 0) {
      return
    }

    const mostProminentFace = detectedFaces.reduce((prev, current) => {
      const prevScore = (prev.boundingBox.width * prev.boundingBox.height) * prev.confidence
      const currentScore = (current.boundingBox.width * current.boundingBox.height) * current.confidence
      return currentScore > prevScore ? current : prev
    })

    setFacePersonData((prev) => {
      const newMap = new Map(prev)
      newMap.set(mostProminentFace.id, latestPersonData)
      return newMap
    })

    console.log(`[FaceDetection] Associated "${latestPersonData.name}" with face ${mostProminentFace.id}`)
    setLatestPersonData(null)
  }, [latestPersonData, detectedFaces])

  useEffect(() => {
    const currentFaceIds = new Set(detectedFaces.map(f => f.id))
    setFacePersonData((prev) => {
      const newMap = new Map(prev)
      for (const faceId of newMap.keys()) {
        if (!currentFaceIds.has(faceId)) {
          newMap.delete(faceId)
        }
      }
      return newMap
    })
  }, [detectedFaces])

  useEffect(() => {
    startWebcam()
    connectSSE()

    return () => {
      stopWebcam()
      disconnectSSE()
    }
  }, [])

  const connectSSE = () => {
    try {
      const es = new EventSource(`${INFERENCE_BACKEND_URL}/stream/inference`)

      console.log('[SSE] Connecting to:', `${INFERENCE_BACKEND_URL}/stream/inference`)

      es.onopen = () => {
        console.log('[SSE] Connected')
      }

      es.addEventListener('inference', (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('[SSE] Received inference event:', message)

          if (message.name && message.description && message.relationship) {
            setLatestPersonData({
              name: message.name,
              description: message.description,
              relationship: message.relationship,
              person_id: message.person_id,
            })
          }
        } catch (err) {
          console.error('[SSE] Error parsing message:', err)
        }
      })

      es.onerror = (error) => {
        console.error('[SSE] Error:', error)
      }

      setEventSource(es)
    } catch (err) {
      console.error('[SSE] Connection error:', err)
    }
  }

  const disconnectSSE = () => {
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
  }

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

  const setupWebRTC = async (stream: MediaStream) => {
    try {
      console.log('[WebRTC] Setting up peer connection')
      const pc = new RTCPeerConnection()
      pcRef.current = pc

      stream.getTracks().forEach(track => {
        console.log('[WebRTC] Adding track:', track.kind)
        pc.addTrack(track, stream)
      })

      pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState)
        setIsConnected(pc.connectionState === 'connected')
      }

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      console.log('[WebRTC] Created offer, waiting for ICE gathering...')

      await waitForIceGathering(pc)
      console.log('[WebRTC] ICE gathering complete, sending offer to backend')

      const response = await fetch(`${OFFER_BACKEND_URL}/offer`, {
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

  const startWebcam = async () => {
    try {
      console.log('[Webcam] Requesting media access...')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      })

      if (videoRef.current) {
        const video = videoRef.current

        const handleMetadataLoaded = () => {
          console.log('[Webcam] Video metadata loaded:', {
            width: video.videoWidth,
            height: video.videoHeight
          })
          setIsVideoReady(true)
        }

        video.addEventListener('loadedmetadata', handleMetadataLoaded)

        if (video.videoWidth > 0 && video.videoHeight > 0) {
          handleMetadataLoaded()
        }

        video.srcObject = stream
        streamRef.current = stream
        stream.getAudioTracks().forEach((track) => {
          track.enabled = !isMuted
        })
        setIsStreaming(true)
        console.log('[Webcam] Stream started')

        setTimeout(() => setupWebRTC(stream), 1000)
      }
    } catch (err) {
      console.error("[Webcam] Error accessing webcam:", err)
    }
  }

  const stopWebcam = () => {
    console.log('[Webcam] Stopping stream')

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }

    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }

    streamRef.current = null
    setIsStreaming(false)
    setIsVideoReady(false)
    setIsConnected(false)
    setIsRayBanMode(false)
    setIsMuted(false)
  }

  const toggleMute = useCallback(() => {
    const stream = streamRef.current
    if (!stream) {
      return
    }
    setIsMuted((prev) => {
      const next = !prev
      stream.getAudioTracks().forEach((track) => {
        track.enabled = !next
      })
      return next
    })
  }, [])

  const faceNotifications = detectedFaces.map((face) => {
    const video = videoRef.current
    const overlay = overlayRef.current

    if (!video || !overlay) return null

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    const overlayWidth = overlay.clientWidth
    const overlayHeight = overlay.clientHeight

    if (videoWidth === 0 || videoHeight === 0) return null

    const transform = calculateVideoTransform(
      videoWidth,
      videoHeight,
      overlayWidth,
      overlayHeight
    )

    const overlayBox = mapBoundingBoxToOverlay(
      face.boundingBox,
      transform,
      overlayWidth,
      true
    )

    const position = calculateNotificationPosition(
      overlayBox,
      overlayWidth,
      overlayHeight
    )

    return {
      face,
      position,
    }
  }).filter((n) => n !== null)

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn(
          "absolute inset-0 h-full w-full object-cover transition-all duration-300",
          isRayBanMode ? "scale-[1.08] blur-[13px]" : "scale-100"
        )}
        style={{ transform: 'scaleX(-1)' }}
      />

      {!isRayBanMode && (
        <>
          <div ref={overlayRef} className="absolute inset-0 pointer-events-none">
            {faceNotifications.map((notification) => {
              const person = facePersonData.get(notification!.face.id)
              return (
                <FaceNotification
                  key={notification!.face.id}
                  faceId={notification!.face.id}
                  left={notification!.position.left}
                  top={notification!.position.top}
                  confidence={notification!.face.confidence}
                  name={person?.name}
                  description={person?.description}
                  relationship={person?.relationship}
                />
              )
            })}
          </div>

          <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} ${isConnected ? 'animate-pulse' : ''}`} />
            <span className="text-xs text-white/80">{isConnected ? 'Connected (WebRTC)' : 'Disconnected'}</span>
          </div>

          {isFaceDetectionLoading && (
            <div className="absolute top-4 left-4 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
              <span className="text-xs text-white/80">Loading face detection...</span>
            </div>
          )}
          {faceDetectionError && (
            <div className="absolute top-4 left-4 px-3 py-2 bg-red-500/60 backdrop-blur-sm rounded-full">
              <span className="text-xs text-white/80">Face detection error</span>
            </div>
          )}
        </>
      )}

      <RayBanOverlay stream={streamRef.current} videoRef={videoRef} visible={isRayBanMode} />

      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center px-6 py-4 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="icon"
            variant={isMuted ? "default" : "secondary"}
            onClick={toggleMute}
            className="h-12 w-12 rounded-full"
            disabled={!isStreaming}
            aria-pressed={isMuted}
          >
            {isMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isMuted ? "Unmute" : "Mute"}</span>

          <Button
            size="icon"
            variant={isStreaming ? "default" : "secondary"}
            onClick={isStreaming ? stopWebcam : startWebcam}
            className="h-12 w-12 rounded-full"
          >
            {isStreaming ? <Video className="h-5 w-5" /> : <VideoOff className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isStreaming ? "Stop Video" : "Start Video"}</span>

          <Button
            variant={isRayBanMode ? "default" : "secondary"}
            className="rounded-full px-4"
            disabled={!isStreaming}
            onClick={() => setIsRayBanMode((prev) => !prev)}
          >
            {isRayBanMode ? "Exit Ray-Ban Mode" : "Enter Ray-Ban Mode"}
          </Button>
        </div>
      </div>
    </div>
  )
}
