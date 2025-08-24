// src/pages/CompleteProfile.jsx
import { useState } from 'react';
import { useNavigate, useLocation, useSearchParams } from 'react-router-dom';
import Logo from '../assets/Logo.png';
import PatientProfileForm from '../components/PatientProfileForm';
import DoctorProfileForm from '../components/DoctorProfileForm';

export default function CompleteProfile() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  
  // Get role from state or URL parameter
  const selectedRole = location.state?.role || searchParams.get('role') || 'Patient';

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('userRole'); // clear saved role
    console.log('User logged out');
    navigate('/signin');
  };

  // Handle profile completion
  const handleProfileComplete = () => {
    // Save role in localStorage for dashboard persistence
    localStorage.setItem('userRole', selectedRole);

    // Navigate to role-based landing page
    navigate(`/landing?role=${selectedRole}`);
  };

  return (
    <div className="min-h-screen flex overflow-hidden bg-gray-50">
      {/* Left Side - 65% White background */}
      <div className="hidden lg:flex lg:w-[65%] bg-white items-center justify-center">
        <div className="text-center">
          <img 
            src={Logo} 
            alt="Doctor Appointment System Logo" 
            className="w-40 h-40 mx-auto mb-6 object-contain"
          />
          <h1 className="text-4xl font-bold text-secondary mb-4">Doctor Appointment System</h1>
          <p className="text-xl text-textLight">Complete your profile to get started</p>
        </div>
      </div>

      {/* Right Side - 35% Profile Form */}
      <div className="w-full lg:w-[35%] flex flex-col p-8 bg-gray-50 h-screen">
        <div className="w-full max-w-2xl h-full flex flex-col">
          {/* Mobile Logo */}
          <div className="lg:hidden flex justify-center mb-6">
            <img 
              src={Logo} 
              alt="Logo" 
              className="w-20 h-20 object-contain"
            />
          </div>

          {/* Header with Logout Button */}
          <div className="flex justify-between items-center mb-6">
            <div className="flex-1"></div>
            <button
              onClick={handleLogout}
              className="flex items-center text-red-600 hover:text-red-800 transition-colors font-medium"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              Logout
            </button>
          </div>

          {/* Header */}
          <div className="text-center mb-8 shrink-0">
            <h1 className="text-3xl font-bold text-secondary mb-2">Complete Your Profile</h1>
            <p className="text-textLight">Finish setting up your {selectedRole.toLowerCase()} account</p>
          </div>

          {/* Scrollable Form Container */}
          <div className="flex-1 overflow-y-auto pb-4">
            {/* Conditional Form Rendering */}
            {selectedRole === 'Patient' ? (
              <PatientProfileForm onComplete={handleProfileComplete} />
            ) : (
              <DoctorProfileForm onComplete={handleProfileComplete} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}