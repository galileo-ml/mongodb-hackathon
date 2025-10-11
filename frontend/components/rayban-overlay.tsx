"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useSearchParams } from "next/navigation"
import { calculateVideoTransform } from "@/lib/coordinate-mapper"

interface RayBanOverlayProps {
  stream: MediaStream | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  visible: boolean
}

/**
 * Ray-Ban inspired overlay with movable glasses that reveal the unblurred feed.
 */
export function RayBanOverlay({ stream: _stream, videoRef, visible }: RayBanOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null)
  const glassesRef = useRef<HTMLDivElement>(null)
  const leftLensRef = useRef<HTMLDivElement>(null)
  const rightLensRef = useRef<HTMLDivElement>(null)
  const leftCanvasRef = useRef<HTMLCanvasElement>(null)
  const rightCanvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<HTMLIFrameElement>(null)
  const searchParams = useSearchParams()
  const [isHudVisible, setIsHudVisible] = useState(false)

  const overlayUrl = useMemo(() => {
    if (!visible) return null
    const raw = searchParams?.get("overlay_url")
    if (!raw) return null
    try {
      return new URL(raw, typeof window !== "undefined" ? window.location.origin : undefined).toString()
    } catch (error) {
      console.warn("[RayBanOverlay] Ignoring invalid overlay_url parameter", error)
      return null
    }
  }, [searchParams, visible])

  useEffect(() => {
    setIsHudVisible(Boolean(overlayUrl))
  }, [overlayUrl])

  useEffect(() => {
    if (!visible) return
    const frame = frameRef.current
    if (!frame) return
    frame.src = overlayUrl ?? "about:blank"
  }, [overlayUrl, visible])

  useEffect(() => {
    if (!visible) return
    const frame = frameRef.current
    if (!frame) return

    const handleMessage = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow) {
        return
      }
      const payload = event.data
      if (payload && typeof payload === "object" && "type" in payload) {
        if (payload.type === "panel-visibility") {
          setIsHudVisible(Boolean(payload.visible))
        }
      }
    }

    window.addEventListener("message", handleMessage)
    return () => {
      window.removeEventListener("message", handleMessage)
    }
  }, [visible])

  useEffect(() => {
    if (!visible) return

    let rafId: number

    const render = () => {
      const video = videoRef.current
      const overlay = overlayRef.current
      if (!video || !overlay || video.readyState < 2) {
        rafId = requestAnimationFrame(render)
        return
      }

      const viewportWidth = overlay.clientWidth
      const viewportHeight = overlay.clientHeight

      if (viewportWidth === 0 || viewportHeight === 0) {
        rafId = requestAnimationFrame(render)
        return
      }

      const overlayRect = overlay.getBoundingClientRect()

      const transform = calculateVideoTransform(
        video.videoWidth || 1,
        video.videoHeight || 1,
        viewportWidth,
        viewportHeight
      )

      const drawLens = (lens: HTMLDivElement | null, canvas: HTMLCanvasElement | null) => {
        if (!lens || !canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const rect = lens.getBoundingClientRect()
        const relativeLeft = rect.left - overlayRect.left
        const relativeTop = rect.top - overlayRect.top

        const width = Math.max(1, Math.round(rect.width))
        const height = Math.max(1, Math.round(rect.height))
        if (canvas.width !== width) {
          canvas.width = width
        }
        if (canvas.height !== height) {
          canvas.height = height
        }

        const { scale, offsetX, offsetY, sourceWidth, sourceHeight } = transform

        const clipLeftRaw = (relativeLeft - offsetX) / scale
        const clipTopRaw = (relativeTop - offsetY) / scale
        const clipWidthRaw = rect.width / scale
        const clipHeightRaw = rect.height / scale

        let clipLeft = clipLeftRaw
        let clipTop = clipTopRaw
        let clipWidth = clipWidthRaw
        let clipHeight = clipHeightRaw

        let destX = 0
        let destY = 0
        let destWidth = width
        let destHeight = height

        if (clipLeft < 0) {
          const delta = Math.min(-clipLeft, clipWidth)
          clipLeft = 0
          clipWidth -= delta
          const deltaPx = delta * scale
          destX = deltaPx
          destWidth -= deltaPx
        }

        if (clipTop < 0) {
          const delta = Math.min(-clipTop, clipHeight)
          clipTop = 0
          clipHeight -= delta
          const deltaPx = delta * scale
          destY = deltaPx
          destHeight -= deltaPx
        }

        let mirroredLeft = sourceWidth - clipLeft - clipWidth
        if (mirroredLeft < 0) {
          const delta = Math.min(-mirroredLeft, clipWidth)
          mirroredLeft = 0
          clipWidth -= delta
          const deltaPx = delta * scale
          destWidth -= deltaPx
        }

        const overflowRight = mirroredLeft + clipWidth - sourceWidth
        if (overflowRight > 0) {
          const delta = Math.min(overflowRight, clipWidth)
          clipWidth -= delta
          const deltaPx = delta * scale
          destWidth -= deltaPx
        }

        const overflowBottom = clipTop + clipHeight - sourceHeight
        if (overflowBottom > 0) {
          const delta = Math.min(overflowBottom, clipHeight)
          clipHeight -= delta
          const deltaPx = delta * scale
          destHeight -= deltaPx
        }

        if (
          clipWidth <= 0 ||
          clipHeight <= 0 ||
          destWidth <= 0 ||
          destHeight <= 0 ||
          mirroredLeft >= sourceWidth ||
          clipTop >= sourceHeight
        ) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          return
        }

        destX = Math.max(0, Math.min(destX, width))
        destY = Math.max(0, Math.min(destY, height))
        destWidth = Math.min(destWidth, width - destX)
        destHeight = Math.min(destHeight, height - destY)

        ctx.save()
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)

        const drawX = canvas.width - destX - destWidth

        ctx.drawImage(
          video,
          mirroredLeft,
          clipTop,
          clipWidth,
          clipHeight,
          drawX,
          destY,
          destWidth,
          destHeight
        )

        ctx.restore()
      }

      drawLens(leftLensRef.current, leftCanvasRef.current)
      drawLens(rightLensRef.current, rightCanvasRef.current)

      rafId = requestAnimationFrame(render)
    }

    rafId = requestAnimationFrame(render)

    return () => {
      if (rafId) {
        cancelAnimationFrame(rafId)
      }
    }
  }, [visible, videoRef])

  useEffect(() => {
    if (!visible) return
    const overlay = overlayRef.current
    const glasses = glassesRef.current
    if (!overlay || !glasses) return

    let pointerId: number | null = null
    let offsetX = 0
    let offsetY = 0

    const setPosition = (x: number, y: number) => {
      glasses.style.left = `${x}px`
      glasses.style.top = `${y}px`
    }

    const centerGlasses = () => {
      const centerX = (overlay.clientWidth - glasses.offsetWidth) / 2
      const centerY = (overlay.clientHeight - glasses.offsetHeight) / 2
      setPosition(centerX, centerY)
      glasses.style.opacity = "1"
    }

    centerGlasses()

    const handlePointerDown = (event: PointerEvent) => {
      event.preventDefault()
      if ((event.target as HTMLElement | null)?.closest("[data-lens-hud]")) {
        return
      }

      pointerId = event.pointerId
      glasses.setPointerCapture(pointerId)
      glasses.dataset.dragging = "true"

      offsetX = event.clientX - glasses.offsetLeft
      offsetY = event.clientY - glasses.offsetTop
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (pointerId !== event.pointerId) return
      setPosition(event.clientX - offsetX, event.clientY - offsetY)
    }

    const handlePointerUp = (event: PointerEvent) => {
      if (pointerId !== event.pointerId) return
      glasses.releasePointerCapture(pointerId)
      pointerId = null
      glasses.dataset.dragging = "false"
    }

    const handleResize = () => {
      centerGlasses()
    }

    glasses.addEventListener("pointerdown", handlePointerDown)
    glasses.addEventListener("pointermove", handlePointerMove)
    glasses.addEventListener("pointerup", handlePointerUp)
    glasses.addEventListener("pointercancel", handlePointerUp)
    window.addEventListener("resize", handleResize)

    return () => {
      glasses.dataset.dragging = "false"
      glasses.removeEventListener("pointerdown", handlePointerDown)
      glasses.removeEventListener("pointermove", handlePointerMove)
      glasses.removeEventListener("pointerup", handlePointerUp)
      glasses.removeEventListener("pointercancel", handlePointerUp)
      window.removeEventListener("resize", handleResize)
    }
  }, [visible])

  if (!visible) {
    return null
  }

  return (
    <div ref={overlayRef} className="pointer-events-none fixed inset-0">
      <div
        ref={glassesRef}
        data-dragging="false"
        className="pointer-events-auto absolute flex items-center gap-4 rounded-[90px] bg-black/25 px-8 py-6 shadow-[0_30px_70px_rgba(0,0,0,0.55)] transition-shadow duration-150 touch-action-none cursor-grab data-[dragging=true]:cursor-grabbing opacity-0"
      >
        <div
          ref={leftLensRef}
          className="relative flex h-[330px] w-[510px] items-center justify-center overflow-hidden rounded-[60px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
        >
          <canvas ref={leftCanvasRef} className="h-full w-full" />
        </div>

        <div className="pointer-events-none absolute left-1/2 top-4 z-20 h-8 w-[150px] -translate-x-1/2 rounded-[20px] bg-black shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08),0_6px_16px_rgba(0,0,0,0.45)]" />

        <div
          ref={rightLensRef}
          className="relative flex h-[330px] w-[510px] items-center justify-center overflow-hidden rounded-[60px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
        >
          <canvas ref={rightCanvasRef} className="h-full w-full" />
          {overlayUrl && isHudVisible && (
            <div
              data-lens-hud
              className="pointer-events-auto absolute bottom-4 right-4 flex h-[110px] w-[160px] overflow-hidden rounded-[14px] border border-white/15 bg-black/60 shadow-[0_8px_20px_rgba(0,0,0,0.35)]"
            >
              <iframe
                ref={frameRef}
                title="Overlay module"
                loading="lazy"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                referrerPolicy="strict-origin-when-cross-origin"
                className="h-full w-full border-0 bg-[#040404]"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
