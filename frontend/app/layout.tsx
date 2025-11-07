import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { initializeTheme } from '@/store/theme-store'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Knowledge Hub - AI-Powered Search Platform',
  description: 'Advanced search platform with hybrid search, personalization, and AI-powered insights',
  keywords: ['search', 'AI', 'knowledge management', 'personalization', 'hybrid search'],
  authors: [{ name: 'Knowledge Hub Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' }
  ],
  manifest: '/manifest.json',
  icons: {
    icon: [
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
      { url: '/favicon-96x96.png', sizes: '96x96', type: 'image/png' }
    ],
    apple: [
      { url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' }
    ]
  },
  openGraph: {
    title: 'Knowledge Hub - AI-Powered Search Platform',
    description: 'Advanced search platform with hybrid search, personalization, and AI-powered insights',
    type: 'website',
    locale: 'en_US',
    url: 'https://knowledge-hub.vercel.app',
    siteName: 'Knowledge Hub',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Knowledge Hub - AI-Powered Search Platform'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Knowledge Hub - AI-Powered Search Platform',
    description: 'Advanced search platform with hybrid search, personalization, and AI-powered insights',
    images: ['/og-image.png']
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
    yandex: 'your-yandex-verification-code'
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Initialize theme on client side
  React.useEffect(() => {
    const cleanup = initializeTheme()
    return cleanup
  }, [])

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="light dark" />
        <meta name="format-detection" content="telephone=no" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Knowledge Hub" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      </head>
      <body className={inter.className}>
        <div id="root">
          {children}
        </div>
      </body>
    </html>
  )
}