"use client"

import { useEffect } from "react"
import { Card } from "@/components/ui/card"

interface PersonNotificationProps {
  name: string
  description: string
  relationship: string
  onClose: () => void
  autoDismiss?: boolean
  dismissDelay?: number
}

export function PersonNotification({
  name,
  description,
  relationship,
  onClose,
  autoDismiss = true,
  dismissDelay = 8000,
}: PersonNotificationProps) {
  useEffect(() => {
    if (autoDismiss) {
      const timer = setTimeout(() => {
        onClose()
      }, dismissDelay)

      return () => clearTimeout(timer)
    }
  }, [autoDismiss, dismissDelay, onClose])

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-in slide-in-from-bottom-4 duration-300">
      <Card className="w-96 bg-white/5 backdrop-blur-xs border-transparent shadow-sm">
        <div className="p-5">
          {/* Name */}
          <h3 className="text-lg font-semibold text-white mb-1 truncate">
            {name}
          </h3>

          {/* Relationship badge */}
          <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-400/15 text-blue-100 border border-blue-300/20 mb-2">
            {relationship}
          </div>

          {/* Description */}
          <p className="text-sm text-white/90 leading-relaxed line-clamp-2">
            {description}
          </p>
        </div>
      </Card>
    </div>
  )
}