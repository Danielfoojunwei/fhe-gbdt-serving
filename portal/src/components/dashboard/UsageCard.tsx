/**
 * Usage Card Component - Shows current usage statistics
 */

import { ChartBarIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface Usage {
  predictionsCount: number;
  predictionsLimit: number;
  usagePercentage: number;
  periodStart: string;
  periodEnd: string;
  overageCount?: number;
  overageCostCents?: number;
}

interface UsageCardProps {
  usage?: Usage;
  isLoading?: boolean;
}

export function UsageCard({ usage, isLoading }: UsageCardProps) {
  if (isLoading) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg animate-pulse">
        <div className="p-5">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4" />
          <div className="h-8 bg-gray-200 rounded w-1/2 mb-2" />
          <div className="h-2 bg-gray-200 rounded w-full" />
        </div>
      </div>
    );
  }

  const percentage = usage?.usagePercentage || 0;
  const isWarning = percentage >= 80;
  const isExceeded = percentage >= 100;

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <ChartBarIcon
              className={clsx(
                'h-6 w-6',
                isExceeded ? 'text-red-600' : isWarning ? 'text-yellow-600' : 'text-indigo-600'
              )}
              aria-hidden="true"
            />
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">Predictions This Month</dt>
              <dd className="flex items-baseline">
                <div className="text-2xl font-semibold text-gray-900">
                  {(usage?.predictionsCount || 0).toLocaleString()}
                </div>
                <div className="ml-2 text-sm text-gray-500">
                  / {(usage?.predictionsLimit || 0).toLocaleString()}
                </div>
              </dd>
            </dl>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mt-4">
          <div className="relative">
            <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
              <div
                style={{ width: `${Math.min(percentage, 100)}%` }}
                className={clsx(
                  'shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-500',
                  isExceeded
                    ? 'bg-red-600'
                    : isWarning
                    ? 'bg-yellow-500'
                    : 'bg-indigo-600'
                )}
              />
            </div>
          </div>
          <div className="mt-1 flex justify-between text-xs text-gray-500">
            <span>{percentage.toFixed(1)}% used</span>
            {usage?.overageCount && usage.overageCount > 0 && (
              <span className="text-red-600">
                +{usage.overageCount.toLocaleString()} overage (${((usage.overageCostCents || 0) / 100).toFixed(2)})
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="bg-gray-50 px-5 py-3">
        <div className="text-sm">
          <a href="/billing" className="font-medium text-indigo-600 hover:text-indigo-500">
            View billing details
          </a>
        </div>
      </div>
    </div>
  );
}
