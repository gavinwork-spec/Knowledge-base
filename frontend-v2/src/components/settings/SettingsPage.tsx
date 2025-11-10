import React from 'react';
import { motion } from 'framer-motion';
import { Settings } from 'lucide-react';

const SettingsPage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="manufacturing-content"
    >
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Settings className="w-8 h-8" />
          Settings
        </h1>
        <p className="text-muted-foreground">Configure application preferences and integrations</p>
      </div>

      <div className="text-center py-12 text-muted-foreground">
        Settings and configuration interface will be displayed here
      </div>
    </motion.div>
  );
};

export default SettingsPage;