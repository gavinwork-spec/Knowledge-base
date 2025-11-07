import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Wrench,
  TrendingUp,
  TrendingDown,
  Zap,
  Thermometer,
  Gauge,
  Settings,
  Eye
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { api } from '@/services/api';
import type { ManufacturingSystem, SystemStatus } from '@/types';
import { cn, formatPercentage, formatRelativeTime, getSystemStatusColor } from '@/lib/utils';

interface SystemStatusProps {
  className?: string;
}

export function SystemStatus({ className }: SystemStatusProps) {
  const [systems, setSystems] = useState<ManufacturingSystem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSystem, setSelectedSystem] = useState<string | null>(null);

  useEffect(() => {
    loadSystems();
    const interval = setInterval(loadSystems, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadSystems = async () => {
    try {
      const systemsData = await api.manufacturing.getSystems();
      setSystems(systemsData);
    } catch (error) {
      console.error('Failed to load systems:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: SystemStatus) => {
    switch (status) {
      case SystemStatus.ONLINE:
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case SystemStatus.OFFLINE:
        return <XCircle className="h-4 w-4 text-red-600" />;
      case SystemStatus.MAINTENANCE:
        return <Wrench className="h-4 w-4 text-orange-600" />;
      case SystemStatus.WARNING:
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case SystemStatus.ERROR:
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStatusColor = (status: SystemStatus) => {
    return getSystemStatusColor(status);
  };

  const getEfficiencyColor = (efficiency: number) => {
    if (efficiency >= 90) return 'text-green-600';
    if (efficiency >= 75) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'temperature':
        return <Thermometer className="h-4 w-4" />;
      case 'pressure':
        return <Gauge className="h-4 w-4" />;
      case 'vibration':
        return <Activity className="h-4 w-4" />;
      case 'powerConsumption':
        return <Zap className="h-4 w-4" />;
      default:
        return <Settings className="h-4 w-4" />;
    }
  };

  const getMetricValue = (system: ManufacturingSystem, metric: string) => {
    switch (metric) {
      case 'temperature':
        return `${system.temperature}°C`;
      case 'pressure':
        return `${system.pressure} PSI`;
      case 'vibration':
        return `${system.vibration} Hz`;
      case 'powerConsumption':
        return `${system.powerConsumption} kW`;
      default:
        return 'N/A';
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>System Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-20 bg-muted rounded-lg" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const systemStats = {
    total: systems.length,
    online: systems.filter(s => s.status === SystemStatus.ONLINE).length,
    offline: systems.filter(s => s.status === SystemStatus.OFFLINE).length,
    maintenance: systems.filter(s => s.status === SystemStatus.MAINTENANCE).length,
    warning: systems.filter(s => s.status === SystemStatus.WARNING).length,
    averageEfficiency: systems.length > 0
      ? systems.reduce((sum, s) => sum + s.efficiency, 0) / systems.length
      : 0
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold">{systemStats.total}</div>
            <div className="text-xs text-muted-foreground">Total Systems</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <div className="text-2xl font-bold text-green-600">{systemStats.online}</div>
            </div>
            <div className="text-xs text-muted-foreground">Online</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Wrench className="h-4 w-4 text-orange-600" />
              <div className="text-2xl font-bold text-orange-600">{systemStats.maintenance}</div>
            </div>
            <div className="text-xs text-muted-foreground">Maintenance</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <div className="text-2xl font-bold text-yellow-600">{systemStats.warning}</div>
            </div>
            <div className="text-xs text-muted-foreground">Warning</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <XCircle className="h-4 w-4 text-red-600" />
              <div className="text-2xl font-bold text-red-600">{systemStats.offline}</div>
            </div>
            <div className="text-xs text-muted-foreground">Offline</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-blue-600" />
              <div className="text-2xl font-bold">
                {formatPercentage(systemStats.averageEfficiency)}
              </div>
            </div>
            <div className="text-xs text-muted-foreground">Avg Efficiency</div>
          </CardContent>
        </Card>
      </div>

      {/* Systems List */}
      <div className="grid gap-4">
        {systems.map((system, index) => (
          <motion.div
            key={system.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card className={cn(
              "transition-all duration-200 hover:shadow-md",
              selectedSystem === system.id && "ring-2 ring-primary"
            )}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(system.status)}
                    <div>
                      <CardTitle className="text-lg">{system.name}</CardTitle>
                      <div className="text-sm text-muted-foreground">
                        {system.location} • {system.operator}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <div className={cn(
                      "px-3 py-1 rounded-full text-xs font-medium",
                      getStatusColor(system.status)
                    )}>
                      {system.status.toUpperCase()}
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setSelectedSystem(
                        selectedSystem === system.id ? null : system.id
                      )}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>

              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {/* Production Metrics */}
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Production</div>
                    <div className="text-2xl font-bold">
                      {system.productionRate}
                    </div>
                    <div className="text-xs text-muted-foreground">units/hr</div>
                  </div>

                  {/* Efficiency */}
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Efficiency</div>
                    <div className={cn(
                      "text-2xl font-bold flex items-center space-x-1",
                      getEfficiencyColor(system.efficiency)
                    )}>
                      <span>{formatPercentage(system.efficiency)}</span>
                      {system.efficiency >= 85 ? (
                        <TrendingUp className="h-4 w-4" />
                      ) : (
                        <TrendingDown className="h-4 w-4" />
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {system.efficiency >= 85 ? 'Optimal' : 'Suboptimal'}
                    </div>
                  </div>

                  {/* System Metrics */}
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Temperature</div>
                    <div className="flex items-center space-x-2">
                      <Thermometer className="h-4 w-4 text-muted-foreground" />
                      <span className="text-lg font-semibold">{system.temperature}°C</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {system.temperature > 80 ? 'High' : system.temperature < 20 ? 'Low' : 'Normal'}
                    </div>
                  </div>

                  {/* Power Consumption */}
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Power</div>
                    <div className="flex items-center space-x-2">
                      <Zap className="h-4 w-4 text-muted-foreground" />
                      <span className="text-lg font-semibold">{system.powerConsumption} kW</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {system.powerConsumption > 100 ? 'High' : 'Normal'}
                    </div>
                  </div>
                </div>

                {/* Expanded Details */}
                {selectedSystem === system.id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 pt-4 border-t"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <Gauge className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Pressure</span>
                        </div>
                        <div className="text-lg font-semibold">{system.pressure} PSI</div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <Activity className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Vibration</span>
                        </div>
                        <div className="text-lg font-semibold">{system.vibration} Hz</div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <Clock className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Last Maintenance</span>
                        </div>
                        <div className="text-sm">
                          {formatRelativeTime(system.lastMaintenance)}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <Wrench className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Next Maintenance</span>
                        </div>
                        <div className="text-sm">
                          {formatRelativeTime(system.nextMaintenance)}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 flex justify-end space-x-2">
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
                      <Button size="sm">
                        Schedule Maintenance
                      </Button>
                    </div>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {systems.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Systems Found</h3>
            <p className="text-muted-foreground">
              No manufacturing systems are currently configured.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default SystemStatus;