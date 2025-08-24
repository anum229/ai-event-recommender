// src/pages/LandingPage.jsx
import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Logo from '../assets/Logo.png';

export default function LandingPage() {
  const [userRole, setUserRole] = useState('');
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  useEffect(() => {
    // First, check if role is stored in localStorage
    const storedRole = localStorage.getItem('userRole');
    const roleFromUrl = searchParams.get('role');
    const role = storedRole || roleFromUrl || 'Patient';

    // If no role at all, redirect to signin
    if (!role) {
      navigate('/signin');
      return;
    }

    setUserRole(role);
    // Store role in localStorage
    localStorage.setItem('userRole', role);
  }, [searchParams, navigate]);

  const handleLogout = () => {
    localStorage.removeItem('userRole');
    navigate('/signin'); // Ensure login route matches your app
  };

  const getWelcomeMessage = () => {
    switch(userRole) {
      case 'Admin':
        return 'Manage the System';
      case 'Doctor':
        return 'Manage Your Appointments';
      case 'Patient':
        return 'Book Appointments with Ease';
      default:
        return 'Book Appointments with Ease';
    }
  };

  const getDashboardOptions = () => {
    switch(userRole) {
      case 'Admin':
        return [
          { title: 'User Management', description: 'Manage all users in the system', icon: '👥' },
          { title: 'Appointment Stats', description: 'View system-wide appointment statistics', icon: '📊' },
          { title: 'System Settings', description: 'Configure application settings', icon: '⚙️' }
        ];
      case 'Doctor':
        return [
          { title: 'My Appointments', description: 'View and manage your appointments', icon: '📅' },
          { title: 'Patient Records', description: 'Access patient medical records', icon: '📋' },
          { title: 'Availability', description: 'Set your working hours and availability', icon: '⏰' }
        ];
      case 'Patient':
        return [
          { title: 'Book Appointment', description: 'Schedule a new appointment with a doctor', icon: '➕' },
          { title: 'My Appointments', description: 'View your upcoming and past appointments', icon: '📅' },
          { title: 'Medical History', description: 'Access your medical records and history', icon: '📋' }
        ];
      default:
        return [];
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div className="flex items-center">
            <img src={Logo} alt="Logo" className="h-8 w-auto mr-3" />
            <h1 className="text-xl font-semibold text-gray-900">Doctor Appointment System</h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">Logged in as: {userRole}</span>
            <button
              onClick={handleLogout}
              className="bg-red-100 text-red-700 px-4 py-2 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Welcome, {userRole}!
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            {getWelcomeMessage()}
          </p>
        </div>

        {/* Dashboard Options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {getDashboardOptions().map((option, index) => (
            <div key={index} className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-shadow cursor-pointer">
              <div className="text-4xl mb-4">{option.icon}</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{option.title}</h3>
              <p className="text-gray-600">{option.description}</p>
            </div>
          ))}
        </div>

        {/* Recent Activity Section */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-gray-100">
              <div className="flex items-center">
                <div className="bg-blue-100 p-2 rounded-lg mr-3">
                  <span className="text-blue-700">📅</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">Profile Completed</p>
                  <p className="text-xs text-gray-500">Your profile was successfully set up</p>
                </div>
              </div>
              <span className="text-xs text-gray-500">Just now</span>
            </div>
            
            <div className="flex items-center justify-between py-2">
              <div className="flex items-center">
                <div className="bg-green-100 p-2 rounded-lg mr-3">
                  <span className="text-green-700">✅</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">Account Verified</p>
                  <p className="text-xs text-gray-500">Your email address has been verified</p>
                </div>
              </div>
              <span className="text-xs text-gray-500">5 minutes ago</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            © {new Date().getFullYear()} Doctor Appointment System. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}