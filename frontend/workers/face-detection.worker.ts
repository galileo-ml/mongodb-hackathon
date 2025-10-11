/**
 * Web Worker for face detection using MediaPipe
 *
 * This worker runs face detection in a background thread to keep the main thread
 * responsive and avoid blocking the UI during intensive processing.
 */

import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

// Worker state
let faceDetector: FaceDetector | null = null
let isInitialized = false

// Message types
interface InitMessage {
  type: 'init'
  minDetectionConfidence: number
}

interface DetectMessage {
  type: 'detect'
  imageData: ImageData
  timestamp: number
}

interface TerminateMessage {
  type: 'terminate'
}

type WorkerMessage = InitMessage | DetectMessage | TerminateMessage

// Initialize MediaPipe FaceDetector
async function initialize(minDetectionConfidence: number) {
  try {
    console.log('[FaceDetectionWorker] Initializing...')

    // Load MediaPipe vision tasks
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    )

    // Create FaceDetector
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
        delegate: 'GPU',
      },
      runningMode: 'IMAGE', // Use IMAGE mode in worker since we're processing individual frames
      minDetectionConfidence,
    })

    isInitialized = true
    console.log('[FaceDetectionWorker] Initialized successfully')

    // Send success message
    self.postMessage({
      type: 'initialized',
      success: true,
    })
  } catch (error) {
    console.error('[FaceDetectionWorker] Initialization error:', error)
    const errorMsg = error instanceof Error ? error.message : 'Unknown error'

    self.postMessage({
      type: 'initialized',
      success: false,
      error: errorMsg,
    })
  }
}

// Process a single frame
function detectFaces(imageData: ImageData, timestamp: number) {
  if (!faceDetector || !isInitialized) {
    console.warn('[FaceDetectionWorker] Detector not initialized')
    return
  }

  try {
    // Run detection on the image data
    const detectionResult = faceDetector.detect(imageData)

    // Convert detections to serializable format
    const detections = detectionResult.detections.map((d) => ({
      boundingBox: {
        originX: d.boundingBox?.originX ?? 0,
        originY: d.boundingBox?.originY ?? 0,
        width: d.boundingBox?.width ?? 0,
        height: d.boundingBox?.height ?? 0,
      },
      confidence: d.categories[0]?.score ?? 0,
    }))

    // Send results back to main thread
    self.postMessage({
      type: 'detections',
      detections,
      timestamp,
    })
  } catch (error) {
    console.error('[FaceDetectionWorker] Detection error:', error)
    self.postMessage({
      type: 'error',
      error: error instanceof Error ? error.message : 'Detection failed',
      timestamp,
    })
  }
}

// Cleanup
function terminate() {
  if (faceDetector) {
    faceDetector.close()
    faceDetector = null
  }
  isInitialized = false
  console.log('[FaceDetectionWorker] Terminated')
}

// Handle messages from main thread
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data

  switch (message.type) {
    case 'init':
      initialize(message.minDetectionConfidence)
      break

    case 'detect':
      detectFaces(message.imageData, message.timestamp)
      break

    case 'terminate':
      terminate()
      break

    default:
      console.warn('[FaceDetectionWorker] Unknown message type:', message)
  }
}

// Handle errors
self.onerror = (error) => {
  console.error('[FaceDetectionWorker] Worker error:', error)
  self.postMessage({
    type: 'error',
    error: error.message || 'Worker error',
  })
}
