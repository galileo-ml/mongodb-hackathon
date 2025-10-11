"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"

interface FaceNotificationProps {
  faceId: string
  name?: string
  description?: string
  relationship?: string
  left: number
  top: number
  confidence: number
  autoDismiss?: boolean
  dismissDelay?: number
  onClose?: () => void
}

export function FaceNotification({
  faceId,
  name,
  description,
  relationship,
  left,
  top,
  confidence,
  autoDismiss = false,
  dismissDelay = 8000,
  onClose,
}: FaceNotificationProps) {
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    if (autoDismiss && onClose) {
      const timer = setTimeout(() => {
        setDismissed(true)
        onClose()
      }, dismissDelay)

      return () => clearTimeout(timer)
    }
  }, [autoDismiss, dismissDelay, onClose])

  if (dismissed || !name) return null

  return (
    <div
      className="absolute z-40 animate-in slide-in-from-left-2 duration-300"
      style={{
        left: `${left}px`,
        top: `${top}px`,
      }}
    >
      <Card className="w-64 bg-white/5 backdrop-blur-xs border-transparent shadow-sm">
        <div className="p-3">
          <h4 className="text-sm font-semibold text-white mb-1 truncate">
            {name}
          </h4>

          {relationship && (
            <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-400/15 text-blue-100 border border-blue-300/20 mb-1.5">
              {relationship}
            </div>
          )}

          {description && (
            <p className="text-xs text-white/90 leading-relaxed line-clamp-2">
              {description}
            </p>
          )}
        </div>
      </Card>
    </div>
  )
}
