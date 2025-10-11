"use client"

import { useEffect, useState } from "react"
import { X, User } from "lucide-react"

interface PersonInfo {
  name: string
  relationship: string
  description: string
}

export function PersonNotification() {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const interval = setInterval(() => {
      setIsVisible(true)
      // Auto-hide after 8 seconds
      setTimeout(() => setIsVisible(false), 8000)
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const personInfo: PersonInfo = {
    name: "John",
    relationship: "Second son",
    description: "i met this person last week",
  }

  if (!isVisible) return null

  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-50 animate-in slide-in-from-bottom-4 duration-300">
      <div className="relative bg-black/40 backdrop-blur-xl border border-white/10 rounded-lg shadow-2xl p-4 min-w-[400px] max-w-md">
        {/* Close button */}
        <button
          onClick={() => setIsVisible(false)}
          className="absolute top-3 right-3 text-white/60 hover:text-white/90 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>

        {/* Content */}
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 h-10 w-10 rounded-full bg-accent/20 flex items-center justify-center">
            <User className="h-5 w-5 text-accent" />
          </div>

          <div className="flex-1 space-y-1">
            <div className="flex items-baseline gap-2">
              <h3 className="text-white font-semibold text-base">{personInfo.name}</h3>
              <span className="text-white/50 text-sm">{personInfo.relationship}</span>
            </div>
            <p className="text-white/70 text-sm leading-relaxed">{personInfo.description}</p>
          </div>
        </div>
      </div>
    </div>
  )
}