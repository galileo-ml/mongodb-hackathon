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
      <div className="flex items-center gap-3 mb-3">
        <h4 className="text-3xl font-bold text-white bg-gray-800/60 px-4 py-2 rounded-lg backdrop-blur-sm">
          {name}
        </h4>

        {relationship && (
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg text-base font-bold bg-amber-500 text-white shadow-lg">
            {relationship}
          </div>
        )}
      </div>

      {description && (
        <Card className="w-[29rem] bg-gray-900/40 backdrop-blur-xs border-gray-700/30 shadow-sm">
          <div className="px-5 py-1">
            <p className="text-base text-white/90 leading-relaxed line-clamp-3">
              {description}
            </p>
          </div>
        </Card>
      )}
    </div>
  )
}
