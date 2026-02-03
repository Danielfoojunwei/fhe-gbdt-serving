/**
 * Recent Predictions Component
 */

import { ClockIcon } from '@heroicons/react/24/outline';
import { formatDistanceToNow } from 'date-fns';

interface Prediction {
  id: string;
  modelName: string;
  latencyMs: number;
  timestamp: string;
  status: 'success' | 'error';
}

// Mock data for demonstration
const mockPredictions: Prediction[] = [
  { id: '1', modelName: 'fraud-detector', latencyMs: 62, timestamp: new Date().toISOString(), status: 'success' },
  { id: '2', modelName: 'credit-scorer', latencyMs: 58, timestamp: new Date(Date.now() - 300000).toISOString(), status: 'success' },
  { id: '3', modelName: 'churn-predictor', latencyMs: 71, timestamp: new Date(Date.now() - 600000).toISOString(), status: 'success' },
];

export function RecentPredictions() {
  const predictions = mockPredictions;

  return (
    <div className="bg-white shadow rounded-lg">
      <div className="px-4 py-5 sm:px-6 flex items-center justify-between">
        <div className="flex items-center">
          <ClockIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
          <h3 className="ml-2 text-base font-semibold leading-6 text-gray-900">
            Recent Predictions
          </h3>
        </div>
        <a href="/analytics" className="text-sm font-medium text-indigo-600 hover:text-indigo-500">
          View all
        </a>
      </div>
      <div className="border-t border-gray-200">
        {predictions.length === 0 ? (
          <div className="px-4 py-12 text-center">
            <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-2 text-sm text-gray-500">No predictions yet</p>
            <p className="mt-1 text-sm text-gray-500">
              Make your first encrypted prediction to see activity here.
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {predictions.map((prediction) => (
              <li key={prediction.id} className="px-4 py-4 sm:px-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center min-w-0">
                    <div
                      className={`flex-shrink-0 h-2 w-2 rounded-full ${
                        prediction.status === 'success' ? 'bg-green-400' : 'bg-red-400'
                      }`}
                    />
                    <div className="ml-4 min-w-0 flex-1">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {prediction.modelName}
                      </p>
                      <p className="text-sm text-gray-500">
                        {formatDistanceToNow(new Date(prediction.timestamp), { addSuffix: true })}
                      </p>
                    </div>
                  </div>
                  <div className="ml-4 flex-shrink-0">
                    <span className="text-sm text-gray-500">{prediction.latencyMs}ms</span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
