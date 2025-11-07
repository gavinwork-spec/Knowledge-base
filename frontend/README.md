# Knowledge Hub Frontend

A modern, responsive, and accessible frontend for the Knowledge Hub platform built with Next.js, TypeScript, and Tailwind CSS.

## 🚀 Features

### 🎨 Modern UI/UX Design
- **Responsive Design**: Mobile-first approach that works seamlessly on all devices
- **Dark/Light Theme**: System-aware theme switching with smooth transitions
- **Component Library**: Reusable, accessible components with consistent design patterns
- **Glass Morphism**: Modern visual effects with backdrop blur and transparency
- **Micro-interactions**: Smooth animations and transitions powered by Framer Motion

### 🎯 Search Interface
- **Advanced Search Input**: Auto-complete, suggestions, and keyboard shortcuts (⌘K)
- **Search Results**: Rich result cards with highlighting, metadata, and explanations
- **Real-time Feedback**: Loading states, error handling, and optimistic updates
- **Filtering & Sorting**: Advanced filtering options with intuitive UI controls

### ♿ Accessibility (A11y)
- **WCAG 2.1 AA Compliance**: Full accessibility support with proper ARIA labels
- **Keyboard Navigation**: Complete keyboard accessibility for all interactive elements
- **Screen Reader Support**: Optimized for screen readers with semantic HTML
- **Reduced Motion**: Respects user's motion preferences
- **High Contrast**: Support for high contrast mode

### 🔧 Technical Excellence
- **TypeScript**: Full type safety with comprehensive type definitions
- **Performance**: Optimized bundle size, code splitting, and lazy loading
- **SEO**: Built-in SEO optimization with meta tags and structured data
- **PWA Ready**: Progressive Web App capabilities with offline support

## 🛠️ Tech Stack

### Core Framework
- **Next.js 14**: React framework with App Router
- **React 18**: Modern React with concurrent features
- **TypeScript**: Type-safe JavaScript

### Styling & UI
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animation library
- **Radix UI**: Accessible component primitives
- **Lucide React**: Beautiful icon library

### State Management
- **Zustand**: Lightweight state management
- **React Hook Form**: Form handling with validation
- **SWR**: Data fetching and caching

### Development Tools
- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting
- **Storybook**: Component development and testing

## 📁 Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── globals.css        # Global styles and theme variables
│   ├── layout.tsx         # Root layout component
│   └── page.tsx           # Home page
├── components/             # Reusable UI components
│   ├── ui/                # Base UI components
│   │   ├── button.tsx
│   │   ├── input.tsx
│   │   ├── card.tsx
│   │   ├── badge.tsx
│   │   └── loading.tsx
│   ├── layout/            # Layout components
│   │   ├── layout.tsx
│   │   └── header.tsx
│   └── search/            # Search components
│       ├── search-input.tsx
│       └── search-results.tsx
├── lib/                   # Utility functions
│   └── utils.ts
├── hooks/                 # Custom React hooks
│   └── use-theme.ts
├── store/                 # State management
│   └── theme-store.ts
├── types/                 # TypeScript type definitions
│   └── index.ts
├── styles/                # Additional styles
└── public/                # Static assets
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18.0 or later
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Start the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

Create a `.env.local` file in the root directory:

```env
# API URLs
NEXT_PUBLIC_API_BASE_URL=http://localhost:8006
NEXT_PUBLIC_PERSONALIZED_API_URL=http://localhost:8007
NEXT_PUBLIC_KNOWLEDGE_API_URL=http://localhost:8001

# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_PERSONALIZATION=true

# Theme
NEXT_PUBLIC_DEFAULT_THEME=system
```

## 🎨 Design System

### Theme Configuration

The theme system uses CSS custom properties with automatic light/dark mode support:

```css
:root {
  --background: 0 0% 100%;
  --foreground: 240 10% 3.9%;
  --primary: 240 5.9% 10%;
  --secondary: 240 4.8% 95.9%;
  /* ... */
}

.dark {
  --background: 240 10% 3.9%;
  --foreground: 0 0% 98%;
  /* ... */
}
```

### Color Palette

- **Primary**: Main brand color
- **Secondary**: Complementary color
- **Accent**: Emphasis color
- **Muted**: Subtle background colors
- **Destructive**: Error and warning states

### Typography

- **Inter**: Clean, modern sans-serif font
- **JetBrains Mono**: Monospace font for code
- **Responsive sizing**: Scales with viewport width

### Spacing System

- **4px base unit**: Consistent spacing scale
- **Responsive padding**: Adapts to screen size
- **Component spacing**: Standardized margins and padding

## 🧩 Component Library

### Base Components

#### Button
```tsx
import { Button } from '@/components/ui/button'

<Button variant="default" size="lg">
  Click me
</Button>
```

**Variants**: `default`, `destructive`, `outline`, `secondary`, `ghost`, `link`, `gradient`, `glass`
**Sizes**: `sm`, `default`, `lg`, `xl`, `icon`, `icon-sm`, `icon-lg`

#### Input
```tsx
import { Input } from '@/components/ui/input'

<Input
  placeholder="Enter text..."
  leftIcon={<SearchIcon />}
  rightIcon={<LoadingSpinner />}
  error={hasError}
  helperText="Error message"
/>
```

#### Card
```tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

<Card variant="elevated">
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
  </CardHeader>
  <CardContent>
    Card content
  </CardContent>
</Card>
```

**Variants**: `default`, `outlined`, `elevated`, `glass`

### Layout Components

#### Layout
```tsx
import { Layout } from '@/components/layout/layout'

<Layout>
  <div>Page content</div>
</Layout>
```

Features:
- Responsive sidebar navigation
- Mobile menu with smooth transitions
- Theme switching in header
- Search integration

### Search Components

#### SearchInput
```tsx
import { SearchInput } from '@/components/search/search-input'

<SearchInput
  value={query}
  onChange={setQuery}
  onSubmit={handleSearch}
  suggestions={suggestions}
  isLoading={isSearching}
/>
```

Features:
- Auto-complete with keyboard navigation
- Real-time suggestions
- Search history integration
- Loading states

#### SearchResults
```tsx
import { SearchResults } from '@/components/search/search-results'

<SearchResults
  results={results}
  isLoading={isLoading}
  query={query}
  onResultClick={handleResultClick}
  showExplanations
  showMetadata
/>
```

Features:
- Rich result cards with metadata
- Highlighting of search terms
- Explanation of search relevance
- Expertise level indicators

## 🎭 Animations & Transitions

### Framer Motion Integration

Components use Framer Motion for smooth animations:

```tsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.3 }}
>
  Content
</motion.div>
```

### Animation Variants

- **fade-in**: Smooth fade in effect
- **slide-in**: Slide animations from different directions
- **scale-in**: Scale animations with opacity
- **shimmer**: Loading shimmer effect

### Reduced Motion Support

Respects user's motion preferences:
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## 📱 Responsive Design

### Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1023px
- **Desktop**: ≥ 1024px

### Mobile-First Approach

All components are designed mobile-first with progressive enhancement:

```tsx
// Base styles for mobile
<div className="p-4">

// Enhanced for tablet
<div className="p-4 md:p-6">

// Enhanced for desktop
<div className="p-4 md:p-6 lg:p-8">
```

### Responsive Components

- **Navigation**: Collapsible sidebar with mobile menu
- **Search**: Adaptive search input with mobile trigger
- **Cards**: Responsive grid layouts
- **Typography**: Fluid typography scaling

## ♿ Accessibility Features

### ARIA Support

- **Semantic HTML**: Proper use of HTML5 semantic elements
- **ARIA Labels**: Descriptive labels for screen readers
- **Keyboard Navigation**: Full keyboard accessibility
- **Focus Management**: Visible focus indicators

### Screen Reader Support

```tsx
<button
  aria-label="Search knowledge base"
  aria-describedby="search-help"
  aria-expanded={isExpanded}
>
  Search
</button>

<div id="search-help" className="sr-only">
  Use keyboard shortcut ⌘K to open search
</div>
```

### High Contrast Mode

Support for high contrast preferences:
```css
@media (prefers-contrast: high) {
  :root {
    --border: 240 5.9% 0%;
    --muted-foreground: 240 3.8% 0%;
  }
}
```

## 🔧 State Management

### Theme Store

```tsx
import { useThemeStore } from '@/store/theme-store'

const { theme, setTheme, toggleTheme } = useThemeStore()
```

Features:
- Persistent theme preferences
- System theme detection
- Smooth theme transitions
- Reduced motion support

### Component State

Local state management with React hooks:
- useState for component state
- useEffect for side effects
- useCallback for optimized functions
- useMemo for expensive computations

## 🚀 Performance Optimization

### Code Splitting

Automatic code splitting with Next.js:
- Route-based splitting
- Component-level lazy loading
- Dynamic imports for large libraries

### Bundle Optimization

- Tree shaking for unused code
- Image optimization with Next.js Image component
- Font optimization with Next.js font system
- CSS optimization with Tailwind's purge option

### Caching Strategy

- Static asset caching
- API response caching with SWR
- Browser caching headers
- Service Worker for offline support

## 🧪 Testing

### Component Testing

```bash
npm run test
# or
yarn test
```

### Storybook

```bash
npm run storybook
# or
yarn storybook
```

### Type Checking

```bash
npm run type-check
# or
yarn type-check
```

## 📊 Analytics & Monitoring

### Performance Metrics

- Core Web Vitals monitoring
- Bundle size analysis
- Component render performance
- API response time tracking

### Error Tracking

- Client-side error boundaries
- API error handling
- User feedback collection
- Performance monitoring

## 🚀 Deployment

### Build

```bash
npm run build
# or
yarn build
```

### Start Production

```bash
npm run start
# or
yarn start
```

### Environment Variables

Production environment variables:
```env
NODE_ENV=production
NEXT_PUBLIC_API_BASE_URL=https://api.knowledgehub.com
```

## 📚 API Integration

### Search API

```tsx
const searchAPI = {
  async search(query: string) {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    })
    return response.json()
  }
}
```

### Error Handling

Comprehensive error handling with user-friendly messages:
- Network errors
- API validation errors
- Client-side validation
- Fallback UI states

## 🤝 Contributing

### Development Guidelines

1. **Code Style**: Follow Prettier and ESLint rules
2. **Type Safety**: Use TypeScript for all new code
3. **Components**: Make components reusable and accessible
4. **Testing**: Add tests for new features
5. **Documentation**: Update documentation for changes

### Git Workflow

1. Create feature branch from main
2. Make changes with descriptive commits
3. Add tests and documentation
4. Create pull request with description
5. Review and merge

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LobeChat**: For UI/UX inspiration and design patterns
- **shadcn/ui**: For component design system foundation
- **Tailwind CSS**: For utility-first CSS framework
- **Framer Motion**: For smooth animations and transitions
- **Next.js Team**: For the excellent React framework

---

Built with ❤️ by the Knowledge Hub team