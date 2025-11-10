import React from 'react';
import { motion } from 'framer-motion';
import { Target } from 'lucide-react';

const QualityPage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="manufacturing-content"
    >
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Target className="w-8 h-8" />
          Quality Control
        </h1>
        <p className="text-muted-foreground">Quality assurance and inspection management</p>
      </div>

      <div className="text-center py-12 text-muted-foreground">
        Quality control and inspection interface will be displayed here
      </div>
    </motion.div>
  );
};

export default QualityPage;