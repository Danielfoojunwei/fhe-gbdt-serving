/**
 * FHE-GBDT Portal - Dashboard Home Page
 */

import { useSession } from 'next-auth/react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { DashboardLayout } from '../components/layout/DashboardLayout';
import { UsageCard } from '../components/dashboard/UsageCard';
import { ModelsCard } from '../components/dashboard/ModelsCard';
import { RecentPredictions } from '../components/dashboard/RecentPredictions';
import { QuickActions } from '../components/dashboard/QuickActions';
import { useUsage } from '../hooks/useUsage';
import { useModels } from '../hooks/useModels';

export default function Dashboard() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const { data: usage, isLoading: usageLoading } = useUsage();
  const { data: models, isLoading: modelsLoading } = useModels();

  // Redirect to login if not authenticated
  if (status === 'unauthenticated') {
    router.push('/auth/signin');
    return null;
  }

  if (status === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600" />
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Dashboard | FHE-GBDT Portal</title>
        <meta name="description" content="Privacy-preserving ML inference dashboard" />
      </Head>

      <DashboardLayout>
        <div className="space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
            <p className="mt-1 text-sm text-gray-500">
              Welcome back, {session?.user?.name || 'User'}
            </p>
          </div>

          {/* Quick Actions */}
          <QuickActions />

          {/* Stats Grid */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <UsageCard usage={usage} isLoading={usageLoading} />
            <ModelsCard models={models} isLoading={modelsLoading} />
          </div>

          {/* Recent Activity */}
          <RecentPredictions />
        </div>
      </DashboardLayout>
    </>
  );
}
