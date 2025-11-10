// Utility Functions - Manufacturing Knowledge Base

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// CN function for combining Tailwind classes
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Format dates for manufacturing context
export function formatDate(date: string | Date, locale: string = 'en-US'): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat(locale, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(dateObj);
}

// Format time durations for manufacturing operations
export function formatDuration(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

// Format file sizes for document uploads
export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// Get expertise level color classes
export function getExpertiseColorClass(level: string): string {
  const colors = {
    beginner: 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400',
    intermediate: 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400',
    advanced: 'text-purple-600 bg-purple-100 dark:bg-purple-900/20 dark:text-purple-400',
    expert: 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400',
  };
  return colors[level as keyof typeof colors] || colors.intermediate;
}

// Get equipment status color classes
export function getEquipmentStatusClass(status: string): string {
  const classes = {
    operational: 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400',
    maintenance: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400',
    error: 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400',
    offline: 'text-gray-600 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400',
  };
  return classes[status as keyof typeof classes] || classes.offline;
}

// Get quality status color classes
export function getQualityStatusClass(status: string): string {
  const classes = {
    pass: 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400',
    fail: 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400',
    marginal: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400',
  };
  return classes[status as keyof typeof classes] || classes.marginal;
}

// Get safety severity color classes
export function getSafetySeverityClass(severity: string): string {
  const classes = {
    low: 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400',
    medium: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400',
    high: 'text-orange-600 bg-orange-100 dark:bg-orange-900/20 dark:text-orange-400',
    critical: 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400',
  };
  return classes[severity as keyof typeof classes] || classes.medium;
}

// Generate manufacturing context string
export function formatManufacturingContext(context: any): string {
  if (!context) return 'General';

  const parts = [];
  if (context.equipment_type) parts.push(context.equipment_type);
  if (context.user_role) parts.push(context.user_role);
  if (context.facility_id && context.facility_id !== 'default') parts.push(context.facility_id);

  return parts.join(' • ') || 'General';
}

// Calculate search relevance score percentage
export function calculateRelevancePercentage(score: number): number {
  return Math.round(Math.min(Math.max(score * 100, 0), 100));
}

// Get compliance standard display name
export function getComplianceStandardName(standard: string): string {
  const names = {
    'ISO_9001': 'ISO 9001',
    'AS9100': 'AS9100',
    'IATF_16949': 'IATF 16949',
    'OSHA': 'OSHA',
    'ANSI': 'ANSI',
  };
  return names[standard as keyof typeof names] || standard;
}

// Validate manufacturing context
export function validateManufacturingContext(context: any): boolean {
  if (!context) return true; // Empty context is valid

  const validRoles = ['operator', 'engineer', 'quality_inspector', 'safety_officer', 'maintenance_tech'];
  const validTypes = ['cnc_milling', 'cnc_turning', 'grinding', 'measurement', 'assembly'];
  const validProcesses = ['machining', 'inspection', 'assembly', 'maintenance', 'quality_control'];

  if (context.user_role && !validRoles.includes(context.user_role)) return false;
  if (context.equipment_type && !validTypes.includes(context.equipment_type)) return false;
  if (context.process_type && !validProcesses.includes(context.process_type)) return false;

  return true;
}

// Generate unique ID for temporary objects
export function generateId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

// Deep copy objects without losing functions
export function deepCopy<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as unknown as T;
  if (obj instanceof Array) return obj.map(item => deepCopy(item)) as unknown as T;
  if (typeof obj === 'object') {
    const copied = {} as T;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        copied[key] = deepCopy(obj[key]);
      }
    }
    return copied;
  }
  return obj;
}

// Debounce function for search input
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
}

// Local storage helpers with error handling
export const storage = {
  get: (key: string): any | null => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return null;
    }
  },

  set: (key: string, value: any): boolean => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error('Error writing to localStorage:', error);
      return false;
    }
  },

  remove: (key: string): boolean => {
    try {
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.error('Error removing from localStorage:', error);
      return false;
    }
  },

  clear: (): boolean => {
    try {
      localStorage.clear();
      return true;
    } catch (error) {
      console.error('Error clearing localStorage:', error);
      return false;
    }
  },
};

// URL helpers for API endpoints
export const apiUrls = {
  knowledge: 'http://localhost:8001',
  chat: 'http://localhost:8002',
  advancedRag: 'http://localhost:8003',
  multiAgent: 'http://localhost:8004',
  reminders: 'http://localhost:8005',
  unifiedSearch: 'http://localhost:8006',
  personalizedSearch: 'http://localhost:8007',
};

// Error message helpers
export function getErrorMessage(error: any): string {
  if (typeof error === 'string') return error;
  if (error?.message) return error.message;
  if (error?.error?.message) return error.error.message;
  return 'An unexpected error occurred';
}

// Success message helpers
export function getSuccessMessage(data: any): string {
  if (typeof data === 'string') return data;
  if (data?.message) return data.message;
  if (data?.data?.message) return data.data.message;
  return 'Operation completed successfully';
}

// Download helper for files
export function downloadFile(data: Blob, filename: string): void {
  const url = URL.createObjectURL(data);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Copy to clipboard helper
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    const success = document.execCommand('copy');
    document.body.removeChild(textArea);
    return success;
  }
}

// Keyboard shortcuts helper
export function getKeyboardShortcut(keys: string[]): string {
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  return keys.map(key => {
    if (key === 'Ctrl' || key === 'Cmd') return isMac ? '⌘' : 'Ctrl';
    if (key === 'Alt') return isMac ? '⌥' : 'Alt';
    if (key === 'Shift') return '⇧';
    return key;
  }).join(isMac ? '' : '+');
}

// Manufacturing-specific validation functions
export function validateTolerance(value: number, tolerance: { min: number; max: number }): boolean {
  return value >= tolerance.min && value <= tolerance.max;
}

export function calculateQualityScore(measurements: Array<{ value: number; tolerance: { min: number; max: number } }>): number {
  if (measurements.length === 0) return 0;

  const validMeasurements = measurements.filter(m => validateTolerance(m.value, m.tolerance));
  return (validMeasurements.length / measurements.length) * 100;
}

export default {
  cn,
  formatDate,
  formatDuration,
  formatFileSize,
  getExpertiseColorClass,
  getEquipmentStatusClass,
  getQualityStatusClass,
  getSafetySeverityClass,
  formatManufacturingContext,
  calculateRelevancePercentage,
  getComplianceStandardName,
  validateManufacturingContext,
  generateId,
  deepCopy,
  debounce,
  storage,
  apiUrls,
  getErrorMessage,
  getSuccessMessage,
  downloadFile,
  copyToClipboard,
  getKeyboardShortcut,
  validateTolerance,
  calculateQualityScore,
};