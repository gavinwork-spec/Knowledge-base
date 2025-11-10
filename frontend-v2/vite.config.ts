import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Development server configuration
  server: {
    port: 3000,
    host: true,
    // API proxy configuration to preserve existing backend endpoints
    proxy: {
      '/api/knowledge': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/knowledge/, '/api'),
      },
      '/api/chat': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/chat/, '/api'),
        ws: true, // Enable WebSocket proxy for chat
      },
      '/api/advanced-rag': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/advanced-rag/, '/api'),
      },
      '/api/multi-agent': {
        target: 'http://localhost:8004',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/multi-agent/, '/api'),
      },
      '/api/reminders': {
        target: 'http://localhost:8005',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/reminders/, '/api'),
      },
      '/api/unified-search': {
        target: 'http://localhost:8006',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/unified-search/, '/api'),
      },
      '/api/personalized-search': {
        target: 'http://localhost:8007',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/personalized-search/, '/api'),
      },
    },
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['framer-motion', '@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
          charts: ['d3'],
        },
      },
    },
  },

  // Path resolution
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@lib': path.resolve(__dirname, './src/lib'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@types': path.resolve(__dirname, './src/types'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/lib/utils'),
    },
  },

  // Environment variables
  envPrefix: 'VITE_',

  // CSS configuration
  css: {
    devSourcemap: true,
  },

  // Optimization
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'zustand',
      'axios',
      '@tanstack/react-query',
      'framer-motion',
      'lucide-react',
      'clsx',
      'tailwind-merge',
    ],
  },

  // Define global constants
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },
})