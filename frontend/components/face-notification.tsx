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
      className="absolute z-40 animate-in slide-in-from-bottom-2 fade-in duration-300"
      style={{
        left: `${left}px`,
        top: `${top}px`,
      }}
    >
      <div className="flex items-center gap-4 mb-4">
        <h4 className="text-4xl font-bold text-white bg-teal-700/70 px-5 py-2.5 rounded-lg backdrop-blur-sm shadow-lg">
          {name}
        </h4>

        {relationship && (
          <div className="inline-flex items-center px-4 py-2 rounded-lg text-lg font-bold bg-cyan-500 text-white shadow-lg">
            {relationship}
          </div>
        )}
      </div>

      {description && (
        <Card className="w-[35rem] bg-teal-900/30 backdrop-blur-xs border-teal-700/40 shadow-sm">
          <div className="px-6 py-1.5">
            <p className="text-2xl text-white/90 leading-relaxed line-clamp-3">
              {description}
            </p>
          </div>
        </Card>
      )}
    </div>
  )
}
