const { createServer } = require('http')
const { parse } = require('url')
const next = require('next')
const { WebSocketServer } = require('ws')

const dev = process.env.NODE_ENV !== 'production'
const hostname = 'localhost'
const port = parseInt(process.env.PORT || '3000', 10)

const app = next({ dev, hostname, port })
const handle = app.getRequestHandler()

app.prepare().then(() => {
  const server = createServer(async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true)
      await handle(req, res, parsedUrl)
    } catch (err) {
      console.error('Error occurred handling', req.url, err)
      res.statusCode = 500
      res.end('Internal server error')
    }
  })

  // Create WebSocket server
  const wss = new WebSocketServer({ noServer: true })

  // Handle WebSocket upgrade (forward non-/api/ws to Next for HMR)
  const nextUpgradeHandler = app.getUpgradeHandler()
  server.on('upgrade', (request, socket, head) => {
    const { pathname } = parse(request.url)

    if (pathname === '/api/ws') {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request)
      })
    } else {
      // Let Next.js handle upgrades like /_next/webpack-hmr
      nextUpgradeHandler(request, socket, head)
    }
  })

  // Mock AI responses
  const mockResponses = [
    "I can see you're in a well-lit room",
    "Analyzing your surroundings...",
    "Video quality looks great!",
    "Processing frame data",
    "I notice movement in the frame",
    "Clear image received",
    "Analyzing visual context",
    "Frame processed successfully"
  ]

  // Handle WebSocket connections
  wss.on('connection', (ws) => {
    console.log('[WebSocket] Client connected')

    ws.on('message', (data) => {
      try {
        // Check if it's binary data (video frame)
        if (data instanceof Buffer) {
          console.log('[WebSocket] Received frame:', {
            size: data.length,
            timestamp: new Date().toISOString()
          })

          // Send a mock AI response after a short delay
          setTimeout(() => {
            const response = mockResponses[Math.floor(Math.random() * mockResponses.length)]
            ws.send(JSON.stringify({
              type: 'ai-response',
              data: {
                llmResponse: response,
                timestamp: new Date().toISOString()
              }
            }))
          }, 500)
        } else {
          // Handle text messages (if any)
          const message = JSON.parse(data.toString())
          console.log('[WebSocket] Received message:', message)
        }
      } catch (error) {
        console.error('[WebSocket] Error processing message:', error)
      }
    })

    ws.on('close', () => {
      console.log('[WebSocket] Client disconnected')
    })

    ws.on('error', (error) => {
      console.error('[WebSocket] Error:', error)
    })

    // Send connection confirmation
    ws.send(JSON.stringify({
      type: 'connection',
      data: { message: 'Connected to WebSocket server' }
    }))
  })

  server.listen(port, (err) => {
    if (err) throw err
    console.log(`> Ready on http://${hostname}:${port}`)
    console.log(`> WebSocket server ready on ws://${hostname}:${port}/api/ws`)
  })
})
