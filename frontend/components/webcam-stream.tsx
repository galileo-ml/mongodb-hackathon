"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Mic, MicOff, Video, VideoOff } from "lucide-react"
import { AiOverlay } from "@/components/ai-overlay"

export default function WebcamStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isSendingFrames, setIsSendingFrames] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [aiResponse, setAiResponse] = useState<string | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    startWebcam()
    connectWebSocket()

    return () => {
      stopWebcam()
      disconnectWebSocket()
    }
  }, [])

  // Connect to WebSocket
  const connectWebSocket = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/api/ws`

      console.log('[WebSocket] Connecting to:', wsUrl)
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('[WebSocket] Connected')
        setWsConnected(true)
        // Clear any pending reconnect attempts
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = null
        }
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('[WebSocket] Received:', message)

          if (message.type === 'ai-response' && message.data.llmResponse) {
            setAiResponse(message.data.llmResponse)
          }
        } catch (err) {
          console.error('[WebSocket] Error parsing message:', err)
        }
      }

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
      }

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected')
        setWsConnected(false)
        wsRef.current = null

        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('[WebSocket] Attempting to reconnect...')
          connectWebSocket()
        }, 3000)
      }

      wsRef.current = ws
    } catch (err) {
      console.error('[WebSocket] Connection error:', err)
    }
  }

  // Disconnect WebSocket
  const disconnectWebSocket = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setWsConnected(false)
  }

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        setTimeout(() => startStreaming(), 1000)
      }
    } catch (err) {
      console.error("Error accessing webcam:", err)
    }
  }

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
      stopStreaming()
    }
  }

  // Capture frame and send via WebSocket
  const captureAndSendFrame = async () => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

    const canvas = document.createElement("canvas")
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext("2d")

    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0)
      canvas.toBlob(
        async (blob) => {
          if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            try {
              // Send binary frame data
              const arrayBuffer = await blob.arrayBuffer()
              wsRef.current.send(arrayBuffer)
            } catch (err) {
              console.error("[WebSocket] Error sending frame:", err)
            }
          }
        },
        "image/jpeg",
        0.8,
      )
    }
  }

  // Start streaming frames to backend
  const startStreaming = () => {
    if (!isStreaming) return

    setIsSendingFrames(true)
    // Stream at ~10 FPS (every 100ms) - adjust as needed
    streamIntervalRef.current = setInterval(() => {
      captureAndSendFrame()
    }, 100)
  }

  // Stop streaming frames
  const stopStreaming = () => {
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current)
      streamIntervalRef.current = null
    }
    setIsSendingFrames(false)
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Video Background */}
      <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 h-full w-full object-cover" />

      {/* WebSocket Connection Status */}
      <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
        <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'} ${wsConnected ? 'animate-pulse' : ''}`} />
        <span className="text-xs text-white/80">{wsConnected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {aiResponse && <AiOverlay response={aiResponse} onClose={() => setAiResponse(null)} />}

      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-6 py-4 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex items-center gap-3">
          <Button
            size="icon"
            variant={isMuted ? "secondary" : "default"}
            onClick={() => setIsMuted(!isMuted)}
            className="h-12 w-12 rounded-full"
          >
            {isMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isMuted ? "Unmute" : "Mute"}</span>
        </div>

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

        <Button size="lg" variant="destructive" onClick={stopWebcam} className="rounded-md px-8">
          End
        </Button>
      </div>
    </div>
  )
}
