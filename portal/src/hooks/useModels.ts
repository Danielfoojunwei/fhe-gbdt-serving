/**
 * Models data hook
 */

import useSWR from 'swr';

interface Model {
  id: string;
  name: string;
  libraryType: string;
  status: string;
  createdAt: string;
}

const fetcher = async (url: string): Promise<Model[]> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
};

export function useModels() {
  return useSWR<Model[]>('/api/models', fetcher, {
    refreshInterval: 60000, // Refresh every minute
    revalidateOnFocus: true,
  });
}
