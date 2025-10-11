"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, Sparkles, X } from "lucide-react"

interface AiOverlayProps {
  response: string
  onClose: () => void
}

export function AiOverlay({ response, onClose }: AiOverlayProps) {
  const [activeTab, setActiveTab] = useState<"suggestions" | "followup">("suggestions")
  const [inputValue, setInputValue] = useState("")

  return (
    <div className="absolute inset-0 flex items-center justify-center p-4 pointer-events-none">
      <Card className="w-full max-w-3xl pointer-events-auto bg-black/40 backdrop-blur-xl border-white/10 shadow-2xl">
        {/* Header with close button */}
        <div className="relative p-6 pb-4">
          <Button
            size="icon"
            variant="ghost"
            onClick={onClose}
            className="absolute right-4 top-4 h-8 w-8 rounded-full text-white/60 hover:text-white hover:bg-white/10"
          >
            <X className="h-4 w-4" />
          </Button>

          {/* Blue pill button */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex items-center gap-2 bg-blue-600 text-white px-6 py-2.5 rounded-full text-sm font-medium">
              <Sparkles className="h-4 w-4" />
              What should I say?
            </div>
          </div>

          {/* Searched records label */}
          <div className="flex items-center gap-2 text-white/60 text-sm mb-4">
            <Search className="h-4 w-4" />
            <span>Searched records</span>
          </div>

          {/* AI Response */}
          <div className="text-white text-lg leading-relaxed mb-6">{response}</div>

          {/* Tabs */}
          <div className="flex items-center gap-6 border-b border-white/10 pb-3 mb-4">
            <button
              onClick={() => setActiveTab("suggestions")}
              className={`flex items-center gap-2 text-sm font-medium transition-colors ${
                activeTab === "suggestions" ? "text-white" : "text-white/50 hover:text-white/80"
              }`}
            >
              <Sparkles className="h-4 w-4" />
              What should I say?
            </button>
            <button
              onClick={() => setActiveTab("followup")}
              className={`flex items-center gap-2 text-sm font-medium transition-colors ${
                activeTab === "followup" ? "text-white" : "text-white/50 hover:text-white/80"
              }`}
            >
              <Search className="h-4 w-4" />
              Follow-up questions
            </button>
          </div>

          {/* Input field */}
          <div className="relative">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask to start typing"
              className="bg-white/5 border-white/10 text-white placeholder:text-white/40 h-12 rounded-md pr-10"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 text-xs bg-white/10 text-white/60 rounded">⌘</kbd>
              <kbd className="px-1.5 py-0.5 text-xs bg-white/10 text-white/60 rounded">K</kbd>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
