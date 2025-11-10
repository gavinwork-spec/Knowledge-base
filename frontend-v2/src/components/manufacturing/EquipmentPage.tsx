import React from 'react';
import { motion } from 'framer-motion';
import { Wrench } from 'lucide-react';

const EquipmentPage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="manufacturing-content"
    >
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Wrench className="w-8 h-8" />
          Equipment Management
        </h1>
        <p className="text-muted-foreground">Monitor and manage manufacturing equipment status</p>
      </div>

      <div className="text-center py-12 text-muted-foreground">
        Equipment monitoring and management interface will be displayed here
      </div>
    </motion.div>
  );
};

export default EquipmentPage;