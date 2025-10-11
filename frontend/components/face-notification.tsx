"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

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
      <div className="flex justify-end mb-2 animate-in slide-in-from-bottom-2 duration-500 delay-150">
        <Button
          size="sm"
          className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium px-3 py-1.5 h-auto rounded-md shadow-lg"
        >
          What should I say?
        </Button>
      </div>

      <Card className="w-64 bg-gray-900/40 backdrop-blur-xs border-gray-700/30 shadow-sm">
        <div className="p-3">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-base font-bold text-white bg-white/10 px-2 py-1 rounded inline-block">{name}</h4>

            {relationship && (
              <div className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[9px] font-normal bg-amber-500/10 text-amber-400/80 border border-amber-500/20">
                {relationship}
              </div>
            )}
          </div>

          {description && <p className="text-xs text-white/90 leading-relaxed line-clamp-2 mt-2.5">{description}</p>}
        </div>
      </Card>
    </div>
  )
}
