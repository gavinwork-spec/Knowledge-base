import React from 'react';
import { motion } from 'framer-motion';
import { Shield } from 'lucide-react';

const SafetyPage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="manufacturing-content"
    >
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Shield className="w-8 h-8" />
          Safety Management
        </h1>
        <p className="text-muted-foreground">Safety procedures and incident tracking</p>
      </div>

      <div className="text-center py-12 text-muted-foreground">
        Safety management and incident tracking interface will be displayed here
      </div>
    </motion.div>
  );
};

export default SafetyPage;