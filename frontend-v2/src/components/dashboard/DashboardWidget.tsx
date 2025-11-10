import React from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  CheckCircle,
  Clock,
  Target,
  Shield,
  Wrench,
  Users,
  Activity
} from 'lucide-react';
import { cn } from '@lib/utils';

interface DashboardWidgetProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    value?: number;
    period?: string;
  };
  status?: 'success' | 'warning' | 'error' | 'info' | 'operational' | 'maintenance';
  icon?: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

const DashboardWidget: React.FC<DashboardWidgetProps> = ({
  title,
  value,
  subtitle,
  trend,
  status,
  icon,
  className,
  onClick
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'success':
        return 'border-green-200 bg-green-50 text-green-900 dark:border-green-800 dark:bg-green-950/20 dark:text-green-200';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50 text-yellow-900 dark:border-yellow-800 dark:bg-yellow-950/20 dark:text-yellow-200';
      case 'error':
        return 'border-red-200 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950/20 dark:text-red-200';
      case 'operational':
        return 'border-green-200 bg-green-50 text-green-900 dark:border-green-800 dark:bg-green-950/20 dark:text-green-200';
      case 'maintenance':
        return 'border-orange-200 bg-orange-50 text-orange-900 dark:border-orange-800 dark:bg-orange-950/20 dark:text-orange-200';
      default:
        return 'border-border bg-background text-foreground';
    }
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    switch (trend.direction) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      default:
        return null;
    }
  };

  const getTrendColor = () => {
    if (!trend) return '';
    return trend.direction === 'up' ? 'text-green-600' : 'text-red-600';
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={cn(
        "manufacturing-card cursor-pointer p-6 relative overflow-hidden",
        className
      )}
    >
      {/* Background gradient for manufacturing widgets */}
      <div className={cn(
        "absolute inset-0 opacity-5 bg-gradient-to-br",
        status === 'success' && "from-green-500/20 to-emerald-600/20",
        status === 'warning' && "from-yellow-500/20 to-orange-600/20",
        status === 'error' && "from-red-500/20 to-pink-600/20",
        status === 'operational' && "from-blue-500/20 to-cyan-600/20",
        status === 'maintenance' && "from-orange-500/20 to-amber-600/20"
      )} />

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            {icon && (
              <div className="p-2 rounded-lg bg-background/80 backdrop-blur-sm">
                {icon}
              </div>
            )}
            <h3 className="text-sm font-medium text-foreground">{title}</h3>
          </div>

          {/* Trend indicator */}
          {trend && (
            <div className="flex items-center gap-1">
              {getTrendIcon()}
              <span className={cn("text-xs font-medium", getTrendColor())}>
                {trend.value && `${trend.value > 0 ? '+' : ''}${trend.value}%`}
                {trend.period && ` (${trend.period})`}
              </span>
            </div>
          )}

          {/* Status indicator */}
          {status && (
            <div className="flex items-center gap-1">
              {status === 'success' && <CheckCircle className="w-4 h-4 text-green-600" />}
              {status === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-600" />}
              {status === 'error' && <AlertCircle className="w-4 h-4 text-red-600" />}
              {status === 'operational' && <Target className="w-4 h-4 text-blue-600" />}
              {status === 'maintenance' && <Wrench className="w-4 h-4 text-orange-600" />}
            </div>
          )}
        </div>

        {/* Value */}
        <div className="flex items-baseline gap-2">
          <span className={cn(
            "text-2xl font-bold",
            status === 'error' && "text-red-600",
            status === 'warning' && "text-yellow-600",
            status === 'success' && "text-green-600",
            status === 'operational' && "text-blue-600"
          )}>
            {value}
          </span>
          {subtitle && (
            <span className="text-sm text-muted-foreground">{subtitle}</span>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default DashboardWidget;