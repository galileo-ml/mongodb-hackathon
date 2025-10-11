"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { X, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"

interface NotificationToastProps {
  message: string | null
  onClose: () => void
}

export function NotificationToast({ message, onClose }: NotificationToastProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    if (message) {
      setIsVisible(true)
      // Auto-dismiss after 8 seconds
      const timer = setTimeout(() => {
        handleClose()
      }, 8000)

      return () => clearTimeout(timer)
    }
  }, [message])

  const handleClose = () => {
    setIsVisible(false)
    setTimeout(() => {
      onClose()
    }, 300) // Wait for animation to complete
  }

  if (!message) return null

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 transition-all duration-300 ${
        isVisible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0 pointer-events-none"
      }`}
    >
      <Card className="w-[380px] max-w-[calc(100vw-3rem)] p-4 border-2 border-accent bg-card shadow-2xl">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            <div className="w-8 h-8 rounded-md bg-accent/20 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-accent" />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold mb-1">AI Response</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">{message}</p>
          </div>
          <Button variant="ghost" size="icon" className="flex-shrink-0 h-6 w-6 -mt-1 -mr-1" onClick={handleClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>
      </Card>
    </div>
  )
}
