/**
 * Usage data hook
 */

import useSWR from 'swr';

interface Usage {
  predictionsCount: number;
  predictionsLimit: number;
  usagePercentage: number;
  periodStart: string;
  periodEnd: string;
  overageCount?: number;
  overageCostCents?: number;
}

const fetcher = async (url: string): Promise<Usage> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch usage');
  return res.json();
};

export function useUsage() {
  return useSWR<Usage>('/api/billing/usage', fetcher, {
    refreshInterval: 30000, // Refresh every 30 seconds
    revalidateOnFocus: true,
  });
}
