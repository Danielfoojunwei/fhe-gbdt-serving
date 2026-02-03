/**
 * Models Card Component - Shows registered models summary
 */

import { CpuChipIcon, PlusIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface Model {
  id: string;
  name: string;
  libraryType: string;
  status: string;
  createdAt: string;
}

interface ModelsCardProps {
  models?: Model[];
  isLoading?: boolean;
}

export function ModelsCard({ models, isLoading }: ModelsCardProps) {
  if (isLoading) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg animate-pulse">
        <div className="p-5">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4" />
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-10 bg-gray-200 rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  const modelList = models || [];
  const recentModels = modelList.slice(0, 5);

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <CpuChipIcon className="h-6 w-6 text-indigo-600" aria-hidden="true" />
            <h3 className="ml-2 text-lg font-medium text-gray-900">Models</h3>
          </div>
          <span className="text-sm text-gray-500">{modelList.length} total</span>
        </div>

        <div className="mt-4">
          {recentModels.length === 0 ? (
            <div className="text-center py-6">
              <CpuChipIcon className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-500">No models registered yet</p>
              <a
                href="/models/new"
                className="mt-3 inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200"
              >
                <PlusIcon className="-ml-0.5 mr-2 h-4 w-4" />
                Register Model
              </a>
            </div>
          ) : (
            <ul className="divide-y divide-gray-200">
              {recentModels.map((model) => (
                <li key={model.id} className="py-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center min-w-0">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {model.name}
                        </p>
                        <p className="text-sm text-gray-500">
                          {model.libraryType} · {model.id.slice(0, 8)}
                        </p>
                      </div>
                    </div>
                    <div className="ml-4 flex-shrink-0">
                      <span
                        className={clsx(
                          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                          model.status === 'ready'
                            ? 'bg-green-100 text-green-800'
                            : model.status === 'compiling'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-gray-100 text-gray-800'
                        )}
                      >
                        {model.status}
                      </span>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="bg-gray-50 px-5 py-3">
        <div className="text-sm">
          <a href="/models" className="font-medium text-indigo-600 hover:text-indigo-500">
            View all models
          </a>
        </div>
      </div>
    </div>
  );
}
