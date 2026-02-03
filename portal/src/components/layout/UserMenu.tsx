/**
 * User Menu Component
 */

import { Fragment } from 'react';
import { Menu, Transition } from '@headlessui/react';
import { useSession, signOut } from 'next-auth/react';
import { UserCircleIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';

const userNavigation = [
  { name: 'Your Profile', href: '/settings/profile' },
  { name: 'Settings', href: '/settings' },
  { name: 'API Keys', href: '/settings/api-keys' },
];

export function UserMenu() {
  const { data: session } = useSession();

  return (
    <Menu as="div" className="relative">
      <Menu.Button className="-m-1.5 flex items-center p-1.5">
        <span className="sr-only">Open user menu</span>
        {session?.user?.image ? (
          <img
            className="h-8 w-8 rounded-full bg-gray-50"
            src={session.user.image}
            alt=""
          />
        ) : (
          <UserCircleIcon className="h-8 w-8 text-gray-400" />
        )}
        <span className="hidden lg:flex lg:items-center">
          <span
            className="ml-4 text-sm font-semibold leading-6 text-gray-900"
            aria-hidden="true"
          >
            {session?.user?.name || 'User'}
          </span>
        </span>
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute right-0 z-10 mt-2.5 w-48 origin-top-right rounded-md bg-white py-2 shadow-lg ring-1 ring-gray-900/5 focus:outline-none">
          {userNavigation.map((item) => (
            <Menu.Item key={item.name}>
              {({ active }) => (
                <a
                  href={item.href}
                  className={clsx(
                    active ? 'bg-gray-50' : '',
                    'block px-3 py-1 text-sm leading-6 text-gray-900'
                  )}
                >
                  {item.name}
                </a>
              )}
            </Menu.Item>
          ))}
          <Menu.Item>
            {({ active }) => (
              <button
                onClick={() => signOut()}
                className={clsx(
                  active ? 'bg-gray-50' : '',
                  'block w-full text-left px-3 py-1 text-sm leading-6 text-gray-900'
                )}
              >
                Sign out
              </button>
            )}
          </Menu.Item>
        </Menu.Items>
      </Transition>
    </Menu>
  );
}
