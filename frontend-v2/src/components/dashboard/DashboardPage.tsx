import React from 'react';
import { motion } from 'framer-motion';
import {
  LayoutDashboard,
  Activity,
  Target,
  Shield,
  Wrench,
  Users,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';

// Import components
import DashboardWidget from './DashboardWidget';
import Button from '../ui/button';

// Import stores
import { useManufacturingStore, useEquipmentStore } from '@stores';

const DashboardPage: React.FC = () => {
  const { context } = useManufacturingStore();
  const { equipment, getStatusCounts } = useEquipmentStore();

  // Get equipment status counts
  const statusCounts = getStatusCounts();

  // Navigation handlers
  const navigateToSection = (section: string) => {
    console.log(`Navigate to ${section} section`);
    // In a real implementation, this would use React Router
  };

  const refreshData = () => {
    console.log('Refreshing dashboard data');
    // In a real implementation, this would fetch fresh data
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="manufacturing-content"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <LayoutDashboard className="w-8 h-8" />
            Manufacturing Dashboard
          </h1>
          <p className="text-muted-foreground">
            Real-time operations overview for {context.facility_id || 'Main Facility'}
          </p>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" onClick={refreshData}>
            <Activity className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Manufacturing Context Banner */}
      {context && (
        <div className="manufacturing-card mb-6">
          <div className="flex items-center justify-between">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Equipment Type:</span>
                <div className="font-medium capitalize">{context.equipment_type || 'General'}</div>
              </div>
              <div>
                <span className="text-muted-foreground">User Role:</span>
                <div className="font-medium capitalize">{context.user_role || 'Operator'}</div>
              </div>
              <div>
                <span className="text-muted-foreground">Facility:</span>
                <div className="font-medium">{context.facility_id || 'Default'}</div>
              </div>
              <div>
                <span className="text-muted-foreground">Process:</span>
                <div className="font-medium capitalize">{context.process_type || 'General'}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <DashboardWidget
          title="Equipment Status"
          value={`${statusCounts.operational + statusCounts.maintenance}`}
          subtitle={`${equipment.length} total machines`}
          status="operational"
          trend={{
            direction: 'up',
            value: 95,
            period: 'uptime'
          }}
          icon={<Activity className="w-5 h-5" />}
          onClick={() => navigateToSection('equipment')}
        />

        <DashboardWidget
          title="Quality Score"
          value="98.5%"
          subtitle="First pass yield"
          status="success"
          trend={{
            direction: 'up',
            value: 2.1,
            period: 'this week'
          }}
          icon={<Target className="w-5 h-5" />}
          onClick={() => navigateToSection('quality')}
        />

        <DashboardWidget
          title="Safety Incidents"
          value="0"
          subtitle="Last 30 days"
          status="success"
          trend={{
            direction: 'neutral',
            value: 0,
            period: 'this month'
          }}
          icon={<Shield className="w-5 h-5" />}
          onClick={() => navigateToSection('safety')}
        />

        <DashboardWidget
          title="Active Users"
          value="12"
          subtitle="Currently online"
          status="operational"
          icon={<Users className="w-5 h-5" />}
          onClick={() => navigateToSection('users')}
        />
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <DashboardWidget
          title="Production Efficiency"
          value="87.3%"
          subtitle="vs 85.2% target"
          status="success"
          trend={{
            direction: 'up',
            value: 3.2,
            period: 'from yesterday'
          }}
          icon={<TrendingUp className="w-5 h-5" />}
          onClick={() => navigateToSection('production')}
        />

        <DashboardWidget
          title="Maintenance Schedule"
          value="3"
          subtitle="Due this week"
          status="warning"
          icon={<Wrench className="w-5 h-5" />}
          onClick={() => navigateToSection('maintenance')}
        />

        <DashboardWidget
          title="Compliance Score"
          value="99.8%"
          subtitle="ISO 9001 standard"
          status="success"
          trend={{
            direction: 'up',
            value: 0.3,
            period: 'this quarter'
          }}
          icon={<CheckCircle className="w-5 h-5" />}
          onClick={() => navigateToSection('compliance')}
        />
      </div>

      {/* Equipment Status Overview */}
      <div className="manufacturing-card mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-6 h-6" />
          Equipment Status Overview
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <DashboardWidget
            title="Operational"
            value={statusCounts.operational}
            subtitle="Running normally"
            status="operational"
            icon={<CheckCircle className="w-4 h-4" />}
            onClick={() => navigateToSection('equipment')}
          />

          <DashboardWidget
            title="Maintenance"
            value={statusCounts.maintenance}
            subtitle="Scheduled maintenance"
            status="maintenance"
            icon={<Wrench className="w-4 h-4" />}
            onClick={() => navigateToSection('maintenance')}
          />

          <DashboardWidget
            title="Warning"
            value={statusCounts.warning}
            subtitle="Attention required"
            status="warning"
            icon={<AlertTriangle className="w-4 h-4" />}
            onClick={() => navigateToSection('alerts')}
          />

          <DashboardWidget
            title="Error"
            value={statusCounts.error}
            subtitle="Immediate action needed"
            status="error"
            icon={<AlertTriangle className="w-4 h-4" />}
            onClick={() => navigateToSection('alerts')}
          />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="manufacturing-card">
        <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Button
            variant="safety"
            className="h-auto p-4 flex flex-col items-start gap-2"
            onClick={() => navigateToSection('safety-check')}
          >
            <Shield className="w-6 h-6" />
            <div className="text-left">
              <div className="font-medium">Safety Check</div>
              <div className="text-xs opacity-90">Run safety procedures</div>
            </div>
          </Button>

          <Button
            variant="quality"
            className="h-auto p-4 flex flex-col items-start gap-2"
            onClick={() => navigateToSection('quality-inspection')}
          >
            <Target className="w-6 h-6" />
            <div className="text-left">
              <div className="font-medium">Quality Inspection</div>
              <div className="text-xs opacity-90">Start inspection process</div>
            </div>
          </Button>

          <Button
            variant="equipment"
            className="h-auto p-4 flex flex-col items-start gap-2"
            onClick={() => navigateToSection('maintenance-request')}
          >
            <Wrench className="w-6 h-6" />
            <div className="text-left">
              <div className="font-medium">Maintenance</div>
              <div className="text-xs opacity-90">Request maintenance</div>
            </div>
          </Button>

          <Button
            variant="outline"
            className="h-auto p-4 flex flex-col items-start gap-2 hover-lift"
            onClick={() => navigateToSection('reports')}
          >
            <Clock className="w-6 h-6" />
            <div className="text-left">
              <div className="font-medium">View Reports</div>
              <div className="text-xs text-muted-foreground">Analytics & insights</div>
            </div>
          </Button>
        </div>
      </div>
    </motion.div>
  );
};

export default DashboardPage;