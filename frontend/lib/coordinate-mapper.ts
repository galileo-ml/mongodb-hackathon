/**
 * Coordinate mapping utilities for face detection with object-fit: cover
 *
 * Handles conversion from video pixel coordinates to overlay coordinates,
 * accounting for object-fit: cover scaling and video mirroring.
 */

export interface BoundingBox {
  originX: number
  originY: number
  width: number
  height: number
}

export interface Position {
  left: number
  top: number
}

export interface VideoTransform {
  scale: number
  renderedWidth: number
  renderedHeight: number
  offsetX: number
  offsetY: number
  sourceWidth: number
  sourceHeight: number
}

/**
 * Calculate the transform parameters for a video with object-fit: cover
 */
export function calculateVideoTransform(
  videoWidth: number,
  videoHeight: number,
  overlayWidth: number,
  overlayHeight: number
): VideoTransform {
  // For object-fit: cover, we use max scale to fill the container
  const scale = Math.max(overlayWidth / videoWidth, overlayHeight / videoHeight)

  const renderedWidth = videoWidth * scale
  const renderedHeight = videoHeight * scale

  // Calculate centering offsets
  const offsetX = (overlayWidth - renderedWidth) / 2
  const offsetY = (overlayHeight - renderedHeight) / 2

  return {
    scale,
    renderedWidth,
    renderedHeight,
    offsetX,
    offsetY,
    sourceWidth: videoWidth,
    sourceHeight: videoHeight,
  }
}

/**
 * Map a bounding box from video pixel coordinates to overlay coordinates
 *
 * @param box - Bounding box in video pixel coordinates
 * @param transform - Video transform parameters
 * @param overlayWidth - Width of the overlay container
 * @param isMirrored - Whether the video is mirrored (scaleX(-1))
 */
export function mapBoundingBoxToOverlay(
  box: BoundingBox,
  transform: VideoTransform,
  overlayWidth: number,
  isMirrored = true
): BoundingBox {
  const { scale, offsetX, offsetY } = transform

  let x = box.originX * scale + offsetX
  let y = box.originY * scale + offsetY
  const width = box.width * scale
  const height = box.height * scale

  // If video is mirrored, flip x coordinate
  if (isMirrored) {
    x = overlayWidth - (x + width)
  }

  return {
    originX: x,
    originY: y,
    width,
    height,
  }
}

/**
 * Calculate the position for a notification next to a face
 * Positions notification to the right of the face with some padding
 *
 * @param box - Bounding box in overlay coordinates
 * @param overlayWidth - Width of the overlay container
 * @param overlayHeight - Height of the overlay container
 * @param notificationWidth - Width of the notification (for clamping)
 * @param padding - Padding between face and notification
 */
export function calculateNotificationPosition(
  box: BoundingBox,
  overlayWidth: number,
  overlayHeight: number,
  notificationWidth = 240,
  padding = 8
): Position {
  // Position to the right of the face
  let left = box.originX + box.width + padding
  let top = box.originY

  // Clamp to viewport - if it goes off right edge, position on left instead
  if (left + notificationWidth > overlayWidth) {
    left = Math.max(0, box.originX - notificationWidth - padding)
  }

  // Clamp top to viewport
  top = Math.max(0, Math.min(top, overlayHeight - 100)) // Assume min notification height of 100px

  return { left, top }
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max))
}
