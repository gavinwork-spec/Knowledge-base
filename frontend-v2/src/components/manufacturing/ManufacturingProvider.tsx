import React, { createContext, useContext, ReactNode } from 'react';

// Manufacturing context for global state
interface ManufacturingContextType {
  isInitialized: boolean;
  equipmentTypes: string[];
  complianceStandards: string[];
  userRoles: string[];
}

const ManufacturingContext = createContext<ManufacturingContextType>({
  isInitialized: false,
  equipmentTypes: [],
  complianceStandards: [],
  userRoles: [],
});

export const useManufacturing = () => useContext(ManufacturingContext);

interface ManufacturingProviderProps {
  children: ReactNode;
}

const ManufacturingProvider: React.FC<ManufacturingProviderProps> = ({ children }) => {
  const manufacturingContext: ManufacturingContextType = {
    isInitialized: true,
    equipmentTypes: [
      'cnc_milling',
      'cnc_turning',
      'grinding',
      'measurement',
      'assembly',
      'edm',
      'inspection'
    ],
    complianceStandards: [
      'ISO_9001',
      'AS9100',
      'IATF_16949',
      'OSHA',
      'ANSI'
    ],
    userRoles: [
      'operator',
      'engineer',
      'quality_inspector',
      'safety_officer',
      'maintenance_tech'
    ]
  };

  return (
    <ManufacturingContext.Provider value={manufacturingContext}>
      {children}
    </ManufacturingContext.Provider>
  );
};

export default ManufacturingProvider;