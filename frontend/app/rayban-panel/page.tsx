"use client"

import { useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export default function RaybanPanelPage() {
  useEffect(() => {
    window.parent?.postMessage({ type: "panel-visibility", visible: true }, "*")

    return () => {
      window.parent?.postMessage({ type: "panel-visibility", visible: false }, "*")
    }
  }, [])

  return (
    <Card className="w-full h-full bg-black/80 border-white/10 text-white p-4 flex flex-col justify-between">
      <div>
        <h2 className="text-sm font-semibold">Incoming Call</h2>
        <p className="text-xs text-white/70 mt-1">
          Jamie Rivera would like to sync about the latest Ray-Ban updates.
        </p>
      </div>
      <div className="flex gap-2 text-xs">
        <Button size="sm" className="flex-1">Accept</Button>
        <Button size="sm" variant="secondary" className="flex-1" onClick={() => {
          window.parent?.postMessage({ type: "panel-visibility", visible: false }, "*")
        }}>Dismiss</Button>
      </div>
    </Card>
  )
}
