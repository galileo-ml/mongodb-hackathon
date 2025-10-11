/**
 * Face tracking system using IoU (Intersection over Union) matching
 *
 * Tracks faces across frames by matching current detections to previous ones,
 * assigning stable IDs to each tracked face.
 */

import type { BoundingBox } from './coordinate-mapper'

export interface DetectedFace {
  id: string
  boundingBox: BoundingBox
  confidence: number
  frameCount: number // Number of consecutive frames this face has been tracked
}

export interface Detection {
  boundingBox: BoundingBox
  confidence: number
}

/**
 * Calculate Intersection over Union (IoU) for two bounding boxes
 */
export function calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
  // Calculate intersection rectangle
  const x1 = Math.max(box1.originX, box2.originX)
  const y1 = Math.max(box1.originY, box2.originY)
  const x2 = Math.min(
    box1.originX + box1.width,
    box2.originX + box2.width
  )
  const y2 = Math.min(
    box1.originY + box1.height,
    box2.originY + box2.height
  )

  // Calculate intersection area
  const intersectionWidth = Math.max(0, x2 - x1)
  const intersectionHeight = Math.max(0, y2 - y1)
  const intersectionArea = intersectionWidth * intersectionHeight

  // Calculate union area
  const box1Area = box1.width * box1.height
  const box2Area = box2.width * box2.height
  const unionArea = box1Area + box2Area - intersectionArea

  // Calculate IoU
  return unionArea > 0 ? intersectionArea / unionArea : 0
}

/**
 * Face tracker class that maintains face identities across frames
 */
export class FaceTracker {
  private trackedFaces: Map<string, DetectedFace> = new Map()
  private nextId = 0
  private iouThreshold = 0.3 // Minimum IoU to consider a match
  private maxMissedFrames = 10 // Remove face after this many missed frames

  /**
   * Update tracked faces with new detections
   *
   * @param detections - New face detections from current frame
   * @returns Array of tracked faces with stable IDs
   */
  update(detections: Detection[]): DetectedFace[] {
    const currentFaces = new Map<string, DetectedFace>()
    const unmatchedDetections = [...detections]

    // Try to match new detections with existing tracked faces
    for (const [id, trackedFace] of this.trackedFaces) {
      let bestMatch: { index: number; iou: number } | null = null

      // Find best matching detection
      unmatchedDetections.forEach((detection, index) => {
        const iou = calculateIoU(trackedFace.boundingBox, detection.boundingBox)

        if (iou > this.iouThreshold && (!bestMatch || iou > bestMatch.iou)) {
          bestMatch = { index, iou }
        }
      })

      // If found a match, update the tracked face
      if (bestMatch) {
        const detection = unmatchedDetections[bestMatch.index]
        currentFaces.set(id, {
          id,
          boundingBox: detection.boundingBox,
          confidence: detection.confidence,
          frameCount: trackedFace.frameCount + 1,
        })
        // Remove matched detection
        unmatchedDetections.splice(bestMatch.index, 1)
      }
    }

    // Create new tracked faces for unmatched detections
    unmatchedDetections.forEach((detection) => {
      const id = `face_${this.nextId++}`
      currentFaces.set(id, {
        id,
        boundingBox: detection.boundingBox,
        confidence: detection.confidence,
        frameCount: 1,
      })
    })

    // Update tracked faces
    this.trackedFaces = currentFaces

    return Array.from(currentFaces.values())
  }

  /**
   * Clear all tracked faces
   */
  clear(): void {
    this.trackedFaces.clear()
    this.nextId = 0
  }

  /**
   * Get current tracked faces
   */
  getTrackedFaces(): DetectedFace[] {
    return Array.from(this.trackedFaces.values())
  }
}
