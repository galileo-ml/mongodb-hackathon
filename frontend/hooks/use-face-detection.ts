"use client"

import { useEffect, useRef, useState } from 'react'
import type { DetectedFace, Detection } from '@/lib/face-tracker'
import { FaceTracker } from '@/lib/face-tracker'

export interface UseFaceDetectionOptions {
  enabled?: boolean
  minDetectionConfidence?: number
  targetFps?: number // Target detection FPS (default: 20)
  useWorker?: boolean // Use Web Worker for processing (default: true)
}

export interface UseFaceDetectionResult {
  detectedFaces: DetectedFace[]
  isLoading: boolean
  error: string | null
}

/**
 * Hook for real-time face detection using MediaPipe in a Web Worker
 *
 * @param videoElement - Video element to run detection on
 * @param options - Detection options
 */
export function useFaceDetection(
  videoElement: HTMLVideoElement | null,
  options: UseFaceDetectionOptions = {}
): UseFaceDetectionResult {
  const {
    enabled = true,
    minDetectionConfidence = 0.5,
    targetFps = 20,
    useWorker = true,
  } = options

  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const faceTrackerRef = useRef<FaceTracker>(new FaceTracker())
  const lastVideoTimeRef = useRef(-1)
  const rafIdRef = useRef<number | null>(null)
  const lastDetectionTimeRef = useRef(0)
  const processingRef = useRef(false)

  // Initialize Web Worker
  useEffect(() => {
    if (!useWorker) {
      setIsLoading(false)
      return
    }

    let isMounted = true

    const initializeWorker = async () => {
      try {
        setIsLoading(true)
        setError(null)

        console.log('[FaceDetection] Initializing Web Worker...')

        // Create worker
        const worker = new Worker(
          new URL('../workers/face-detection.worker.ts', import.meta.url),
          { type: 'module' }
        )

        // Handle messages from worker
        worker.onmessage = (event) => {
          const message = event.data

          switch (message.type) {
            case 'initialized':
              if (message.success) {
                console.log('[FaceDetection] Worker initialized successfully')
                if (isMounted) {
                  setIsLoading(false)
                }
              } else {
                console.error('[FaceDetection] Worker initialization failed:', message.error)
                if (isMounted) {
                  setError(message.error)
                  setIsLoading(false)
                }
              }
              break

            case 'detections':
              processingRef.current = false
              if (isMounted) {
                const detections: Detection[] = message.detections
                const trackedFaces = faceTrackerRef.current.update(detections)
                setDetectedFaces(trackedFaces)
              }
              break

            case 'error':
              processingRef.current = false
              console.error('[FaceDetection] Worker error:', message.error)
              break

            default:
              console.warn('[FaceDetection] Unknown message from worker:', message)
          }
        }

        worker.onerror = (error) => {
          console.error('[FaceDetection] Worker error:', error)
          if (isMounted) {
            setError('Worker error occurred')
            setIsLoading(false)
          }
        }

        workerRef.current = worker

        // Create canvas for frame capture
        const canvas = document.createElement('canvas')
        canvasRef.current = canvas

        // Initialize worker
        worker.postMessage({
          type: 'init',
          minDetectionConfidence,
        })
      } catch (err) {
        if (isMounted) {
          const errorMsg = err instanceof Error ? err.message : 'Failed to initialize worker'
          setError(errorMsg)
          setIsLoading(false)
          console.error('[FaceDetection] Initialization error:', err)
        }
      }
    }

    initializeWorker()

    return () => {
      isMounted = false
      if (workerRef.current) {
        workerRef.current.postMessage({ type: 'terminate' })
        workerRef.current.terminate()
        workerRef.current = null
      }
      if (canvasRef.current) {
        canvasRef.current = null
      }
    }
  }, [minDetectionConfidence, useWorker])

  // Detection loop - sends frames to worker for processing
  useEffect(() => {
    if (!enabled || !videoElement || isLoading || !useWorker) {
      return
    }

    const worker = workerRef.current
    const canvas = canvasRef.current

    if (!worker || !canvas) {
      return
    }

    const frameIntervalMs = 1000 / targetFps

    const captureAndDetect = () => {
      if (!videoElement || !worker || !canvas) {
        return
      }

      const now = performance.now()
      const currentVideoTime = videoElement.currentTime

      // Skip if video hasn't progressed or we're detecting too fast
      if (
        currentVideoTime === lastVideoTimeRef.current ||
        now - lastDetectionTimeRef.current < frameIntervalMs
      ) {
        rafIdRef.current = requestAnimationFrame(captureAndDetect)
        return
      }

      // Safety check: Verify video has valid dimensions before processing
      if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        console.warn('[FaceDetection] Video dimensions not ready, skipping frame')
        rafIdRef.current = requestAnimationFrame(captureAndDetect)
        return
      }

      // Skip if worker is still processing previous frame
      if (processingRef.current) {
        rafIdRef.current = requestAnimationFrame(captureAndDetect)
        return
      }

      try {
        // Set canvas dimensions to match video
        if (canvas.width !== videoElement.videoWidth || canvas.height !== videoElement.videoHeight) {
          canvas.width = videoElement.videoWidth
          canvas.height = videoElement.videoHeight
        }

        // Draw current video frame to canvas
        const ctx = canvas.getContext('2d', { willReadFrequently: true })
        if (!ctx) {
          console.error('[FaceDetection] Failed to get canvas context')
          return
        }

        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height)

        // Get image data from canvas
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

        // Send to worker for processing
        processingRef.current = true
        worker.postMessage({
          type: 'detect',
          imageData,
          timestamp: now,
        })

        lastVideoTimeRef.current = currentVideoTime
        lastDetectionTimeRef.current = now
      } catch (err) {
        console.error('[FaceDetection] Frame capture error:', err)
        processingRef.current = false
      }

      rafIdRef.current = requestAnimationFrame(captureAndDetect)
    }

    // Start detection loop
    rafIdRef.current = requestAnimationFrame(captureAndDetect)

    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
        rafIdRef.current = null
      }
    }
  }, [enabled, videoElement, isLoading, targetFps, useWorker])

  // Clear faces when disabled
  useEffect(() => {
    if (!enabled) {
      setDetectedFaces([])
      faceTrackerRef.current.clear()
    }
  }, [enabled])

  return {
    detectedFaces,
    isLoading,
    error,
  }
}
