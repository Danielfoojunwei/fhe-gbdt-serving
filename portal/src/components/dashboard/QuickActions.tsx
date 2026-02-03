/**
 * Quick Actions Component
 */

import {
  PlusIcon,
  KeyIcon,
  PlayIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline';

const actions = [
  {
    title: 'Register Model',
    description: 'Upload a new GBDT model',
    href: '/models/new',
    icon: PlusIcon,
    iconBackground: 'bg-indigo-500',
  },
  {
    title: 'Generate Keys',
    description: 'Create new FHE key pair',
    href: '/keys/generate',
    icon: KeyIcon,
    iconBackground: 'bg-green-500',
  },
  {
    title: 'Try Prediction',
    description: 'Test encrypted inference',
    href: '/playground',
    icon: PlayIcon,
    iconBackground: 'bg-purple-500',
  },
  {
    title: 'View Docs',
    description: 'Read the documentation',
    href: 'https://docs.fhe-gbdt.dev',
    icon: DocumentTextIcon,
    iconBackground: 'bg-gray-500',
    external: true,
  },
];

export function QuickActions() {
  return (
    <div className="bg-white shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h3 className="text-base font-semibold leading-6 text-gray-900">Quick Actions</h3>
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {actions.map((action) => (
            <a
              key={action.title}
              href={action.href}
              target={action.external ? '_blank' : undefined}
              rel={action.external ? 'noopener noreferrer' : undefined}
              className="relative flex items-center space-x-3 rounded-lg border border-gray-300 bg-white px-4 py-4 shadow-sm hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              <div className={`flex-shrink-0 rounded-lg p-2 ${action.iconBackground}`}>
                <action.icon className="h-5 w-5 text-white" aria-hidden="true" />
              </div>
              <div className="min-w-0 flex-1">
                <span className="absolute inset-0" aria-hidden="true" />
                <p className="text-sm font-medium text-gray-900">{action.title}</p>
                <p className="text-sm text-gray-500 truncate">{action.description}</p>
              </div>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
