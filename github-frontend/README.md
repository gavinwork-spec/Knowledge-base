# Manufacturing Knowledge Base - Modern UI

A modern React frontend for the Manufacturing Knowledge Base system, built with LobeChat-inspired UI components.

## 🚀 Features

### Modern UI/UX
- **LobeChat-inspired Design**: Clean, modern interface with smooth animations
- **Dark/Light Theme**: Automatic theme switching with system preference detection
- **Responsive Design**: Mobile-first approach that works on all devices
- **Glass Morphism Effects**: Modern visual design with backdrop blur
- **Smooth Animations**: Framer Motion powered transitions and interactions

### Manufacturing Focus
- **System Status Dashboard**: Real-time monitoring of manufacturing systems
- **AI Chat Interface**: Advanced chat with manufacturing-specific context
- **Production Analytics**: Comprehensive metrics and KPIs
- **Alert Management**: Proactive system notifications
- **Knowledge Base Integration**: Seamless access to manufacturing documentation

### Technical Features
- **TypeScript**: Full type safety throughout the application
- **Modern React**: Hooks-based architecture with functional components
- **State Management**: Zustand for efficient state handling
- **API Integration**: Comprehensive REST API client with error handling
- **Component Library**: Reusable UI components with Radix UI
- **Performance Optimized**: Code splitting and lazy loading

## 🛠️ Technology Stack

- **Frontend**: React 18, TypeScript, Vite
- **UI Framework**: Tailwind CSS with custom design system
- **Components**: Radix UI primitives with custom styling
- **Animations**: Framer Motion for smooth transitions
- **State Management**: Zustand with persistence
- **Data Fetching**: React Query with axios
- **Forms**: React Hook Form with Zod validation
- **Routing**: React Router v6
- **Icons**: Lucide React
- **Build Tool**: Vite

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd github-frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the root directory:

```env
VITE_API_BASE_URL=http://localhost:8001/api
VITE_CHAT_API_BASE_URL=http://localhost:8002/api/v1
VITE_APP_NAME=Manufacturing Knowledge Base
VITE_APP_VERSION=2.0.0
```

### Theme Configuration

The theme system supports three modes:
- `light`: Always use light theme
- `dark`: Always use dark theme
- `system`: Use system preference (default)

## 📱 Development

### Project Structure

```
src/
├── components/           # Reusable components
│   ├── ui/              # Base UI components
│   ├── layout/          # Layout components
│   ├── chat/            # Chat interface components
│   └── dashboard/       # Dashboard components
├── services/            # API services
├── store/              # State management
├── types/              # TypeScript type definitions
├── lib/                # Utility functions
├── hooks/              # Custom React hooks
├── App.tsx             # Main application component
├── main.tsx            # Application entry point
└── index.css           # Global styles
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run type-check` - Run TypeScript type checking

## 🎨 UI Components

The application includes a comprehensive component library:

### Base Components
- `Button` - Flexible button with multiple variants
- `Input` - Form input with validation states
- `Card` - Content container with header and footer
- `ThemeToggle` - Theme switching component

### Layout Components
- `Layout` - Main application layout with sidebar
- `Navigation` - Responsive navigation menu
- `Header` - Application header with search

### Business Components
- `ChatInterface` - Manufacturing-specific chat UI
- `SystemStatus` - Real-time system monitoring
- `ProductionMetrics` - KPI visualization
- `AlertPanel` - System alert management

## 🔄 API Integration

The frontend integrates with two main APIs:

### Knowledge Base API (Port 8001)
- System status and metrics
- Knowledge base entries
- User management
- Analytics data

### Chat API (Port 8002)
- AI-powered chat interface
- Context-aware responses
- Manufacturing-specific queries

## 🎯 Features in Detail

### Chat Interface
- Real-time messaging with AI assistant
- Manufacturing-specific context
- Voice input support (planned)
- File attachment support (planned)
- Message history and sessions
- Quick suggestions and shortcuts

### Dashboard
- Real-time system monitoring
- Production metrics visualization
- Alert management and notifications
- Interactive charts and graphs
- Export capabilities

### Theme System
- Automatic system preference detection
- Manual theme switching
- Persistent theme selection
- Smooth theme transitions
- Manufacturing-specific color schemes

## 🔍 Manufacturing Business Logic

The frontend preserves and enhances all existing manufacturing functionality:

### System Monitoring
- Real-time status updates for all manufacturing systems
- Performance metrics and KPIs
- Maintenance scheduling and tracking
- Alert management and escalation

### Knowledge Integration
- Seamless access to manufacturing documentation
- Context-aware AI responses
- Search and discovery capabilities
- Expert system integration

### User Experience
- Role-based access control
- Personalized dashboards
- Manufacturing-specific workflows
- Mobile-optimized interfaces

## 🚀 Deployment

### Production Build

```bash
# Build the application
npm run build

# The build output will be in the `dist` directory
# You can deploy this to any static hosting service
```

### Environment Setup

For production deployment, ensure:

1. Backend APIs are accessible at configured URLs
2. Environment variables are properly set
3. HTTPS is configured for secure connections
4. Proper CSP headers are configured

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Related Projects

- [Knowledge Base API](../api_server_knowledge/) - Backend API
- [Chat API](../api_chat_interface/) - AI Chat service
- [Observability System](../observability/) - Monitoring and analytics

## 🆘 Troubleshooting

### Common Issues

**Development server won't start**
- Check if port 3000 is available
- Verify all dependencies are installed
- Check for TypeScript errors

**API connection issues**
- Verify backend services are running
- Check API URLs in environment variables
- Ensure CORS is configured properly

**Theme not working**
- Check CSS variable definitions
- Verify Tailwind CSS configuration
- Check browser console for errors

### Performance Optimization

- Enable production builds for deployment
- Use React DevTools Profiler for performance debugging
- Monitor bundle size with built-in analyzer
- Implement code splitting for large components

---

Built with ❤️ for Manufacturing Excellence