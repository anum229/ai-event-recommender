import { useState, useRef } from 'react';

export default function DoctorProfileForm() {
  const [formData, setFormData] = useState({
    // Personal Info
    profilePicture: null,
    firstName: '',
    lastName: '',
    gender: '',
    dob: '',
    phone: '',
    password: '',
    confirmPassword: '',
    
    // Identity Verification
    address: '',
    cnicFront: null,
    cnicBack: null,
    
    // Professional Details
    licenseNumber: '',
    licenseDocument: null,
    degreeCertificate: null,
    experience: '',
    speciality: [],
    bio: '',
    
    // Service Details
    consultationFee: '',
    availableSlots: []
  });

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [profilePreview, setProfilePreview] = useState(null);
  const [cnicFrontPreview, setCnicFrontPreview] = useState(null);
  const [cnicBackPreview, setCnicBackPreview] = useState(null);
  const [licenseDocName, setLicenseDocName] = useState('');
  const [degreeCertName, setDegreeCertName] = useState('');
  const [showAvailabilityModal, setShowAvailabilityModal] = useState(false);
  const [availabilitySlots, setAvailabilitySlots] = useState([]);
  const [newSlot, setNewSlot] = useState({
    day: '',
    startHour: '',
    startMinute: '',
    startPeriod: '',
    endHour: '',
    endMinute: '',
    endPeriod: ''
  });
  
  const profileInputRef = useRef(null);
  const cnicFrontInputRef = useRef(null);
  const cnicBackInputRef = useRef(null);
  const licenseInputRef = useRef(null);
  const degreeInputRef = useRef(null);
  
  const specialties = ['Cardiology', 'Pediatrics', 'Dermatology', 'Neurology', 'Orthopedics', 'Gynecology', 'General Medicine'];
  const daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Doctor profile data:', formData);
    // Handle form submission
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSpecialityChange = (speciality) => {
    const updatedSpecialities = formData.speciality.includes(speciality)
      ? formData.speciality.filter(s => s !== speciality)
      : [...formData.speciality, speciality];
    
    setFormData(prev => ({ ...prev, speciality: updatedSpecialities }));
  };

  const handleFileUpload = (field, file, isImage = false, setPreview = null, setName = null) => {
    if (isImage) {
      // Validate image files (JPEG, JPG, PNG)
      if (!file.type.match('image/jpeg') && !file.type.match('image/jpg') && !file.type.match('image/png')) {
        alert('Please select a valid image file (JPEG, JPG, or PNG)');
        return false;
      }
    } else {
      // Validate PDF files
      if (!file.type.match('application/pdf')) {
        alert('Please select a PDF file');
        return false;
      }
    }
    
    // Check file size (max 5MB for images, 10MB for PDFs)
    const maxSize = isImage ? 5 * 1024 * 1024 : 10 * 1024 * 1024;
    if (file.size > maxSize) {
      alert(`Please select a file smaller than ${isImage ? '5MB' : '10MB'}`);
      return false;
    }
    
    setFormData(prev => ({ ...prev, [field]: file }));
    
    // Set file name for display
    if (setName) {
      setName(file.name);
    }
    
    // Create preview for images
    if (isImage && setPreview) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
    
    return true;
  };

  const triggerFileInput = (inputRef) => {
    inputRef.current?.click();
  };

  const handleAddAvailabilitySlot = () => {
    if (!newSlot.day || !newSlot.startHour || !newSlot.startMinute || !newSlot.startPeriod || 
        !newSlot.endHour || !newSlot.endMinute || !newSlot.endPeriod) {
      alert('Please fill all fields for the time slot');
      return;
    }

    // Convert to 24-hour format for validation
    const convertTo24Hour = (hour, minute, period) => {
      let h = parseInt(hour);
      if (period === 'PM' && h !== 12) h += 12;
      if (period === 'AM' && h === 12) h = 0;
      return `${h.toString().padStart(2, '0')}:${minute}`;
    };

    const startTime24 = convertTo24Hour(newSlot.startHour, newSlot.startMinute, newSlot.startPeriod);
    const endTime24 = convertTo24Hour(newSlot.endHour, newSlot.endMinute, newSlot.endPeriod);

    if (startTime24 >= endTime24) {
      alert('End time must be after start time');
      return;
    }

    const slot = {
      day: newSlot.day,
      startTime: `${newSlot.startHour}:${newSlot.startMinute} ${newSlot.startPeriod}`,
      endTime: `${newSlot.endHour}:${newSlot.endMinute} ${newSlot.endPeriod}`,
      startTime24,
      endTime24
    };
    
    setAvailabilitySlots([...availabilitySlots, slot]);
    setFormData(prev => ({ 
      ...prev, 
      availableSlots: [...prev.availableSlots, slot] 
    }));
    
    // Reset the form
    setNewSlot({
      day: '',
      startHour: '',
      startMinute: '',
      startPeriod: '',
      endHour: '',
      endMinute: '',
      endPeriod: ''
    });
    
    setShowAvailabilityModal(false);
  };

  const handleRemoveAvailabilitySlot = (index) => {
    const updatedSlots = [...availabilitySlots];
    updatedSlots.splice(index, 1);
    setAvailabilitySlots(updatedSlots);
    setFormData(prev => ({ ...prev, availableSlots: updatedSlots }));
  };

  return (
    <>
      <form onSubmit={handleSubmit} className="space-y-6 bg-white p-8 rounded-2xl shadow-lg">
        {/* Personal Info Section */}
        <div className="border-b pb-6">
          <h3 className="text-lg font-semibold text-secondary mb-4">Personal Information</h3>
          
          {/* Profile Picture */}
          <div className="text-center mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Profile Picture <span className="text-red-500">*</span>
            </label>
            <div className="flex justify-center">
              <div 
                className="relative w-24 h-24 border-2 border-dashed border-gray-300 rounded-full flex items-center justify-center cursor-pointer overflow-hidden"
                onClick={() => triggerFileInput(profileInputRef)}
              >
                {profilePreview ? (
                  <img 
                    src={profilePreview} 
                    alt="Profile preview" 
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="flex flex-col items-center">
                    <svg className="w-8 h-8 text-gray-400 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    <span className="text-xs text-textLight">Upload</span>
                  </div>
                )}
                <input
                  type="file"
                  ref={profileInputRef}
                  onChange={(e) => {
                    if (e.target.files[0]) {
                      handleFileUpload('profilePicture', e.target.files[0], true, setProfilePreview);
                    }
                  }}
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                />
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2">JPEG, JPG or PNG (Max 5MB)</p>
          </div>

          {/* Name Fields */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                First Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={formData.firstName}
                onChange={(e) => handleInputChange('firstName', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Last Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={formData.lastName}
                onChange={(e) => handleInputChange('lastName', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
              />
            </div>
          </div>

          {/* Gender - Now on its own line */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Gender <span className="text-red-500">*</span>
            </label>
            <div className="flex space-x-4 mt-2">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="gender"
                  value="Male"
                  checked={formData.gender === 'Male'}
                  onChange={(e) => handleInputChange('gender', e.target.value)}
                  className="w-4 h-4 text-secondary focus:ring-secondary border-gray-300"
                  required
                />
                <span className="text-sm text-gray-700">Male</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="gender"
                  value="Female"
                  checked={formData.gender === 'Female'}
                  onChange={(e) => handleInputChange('gender', e.target.value)}
                  className="w-4 h-4 text-secondary focus:ring-secondary border-gray-300"
                />
                <span className="text-sm text-gray-700">Female</span>
              </label>
            </div>
          </div>

          {/* DOB and Phone - Moved to next line */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                DOB <span className="text-red-500">*</span>
              </label>
              <input
                type="date"
                required
                value={formData.dob}
                onChange={(e) => handleInputChange('dob', e.target.value)}
                max={new Date().toISOString().split('T')[0]}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Phone <span className="text-red-500">*</span>
              </label>
              <input
                type="tel"
                required
                value={formData.phone}
                onChange={(e) => handleInputChange('phone', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
              />
            </div>
          </div>

          {/* Password */}
          <div className="grid grid-cols-2 gap-4">
            <div className="relative">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password <span className="text-red-500">*</span>
              </label>
              <input
                type={showPassword ? 'text' : 'password'}
                required
                value={formData.password}
                onChange={(e) => handleInputChange('password', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary pr-10"
              />
              <button
                type="button"
                className="absolute right-3 top-9 text-gray-500 hover:text-gray-700"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                  </svg>
                )}
              </button>
            </div>
            <div className="relative">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Confirm Password <span className="text-red-500">*</span>
              </label>
              <input
                type={showConfirmPassword ? 'text' : 'password'}
                required
                value={formData.confirmPassword}
                onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary pr-10"
              />
              <button
                type="button"
                className="absolute right-3 top-9 text-gray-500 hover:text-gray-700"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              >
                {showConfirmPassword ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Identity Verification Section */}
        <div className="border-b pb-6">
          <h3 className="text-lg font-semibold text-secondary mb-4">Identity Verification</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Complete Residence <span className="text-red-500">*</span>
            </label>
            <textarea
              required
              value={formData.address}
              onChange={(e) => handleInputChange('address', e.target.value)}
              rows={2}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                CNIC Front <span className="text-red-500">*</span>
              </label>
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer h-32 flex items-center justify-center"
                onClick={() => triggerFileInput(cnicFrontInputRef)}
              >
                {cnicFrontPreview ? (
                  <img 
                    src={cnicFrontPreview} 
                    alt="CNIC Front preview" 
                    className="h-full object-contain"
                  />
                ) : (
                  <span className="text-textLight">Upload Front (Image)</span>
                )}
                <input
                  type="file"
                  ref={cnicFrontInputRef}
                  onChange={(e) => {
                    if (e.target.files[0]) {
                      handleFileUpload('cnicFront', e.target.files[0], true, setCnicFrontPreview);
                    }
                  }}
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">JPEG, JPG or PNG (Max 5MB)</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                CNIC Back <span className="text-red-500">*</span>
              </label>
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer h-32 flex items-center justify-center"
                onClick={() => triggerFileInput(cnicBackInputRef)}
              >
                {cnicBackPreview ? (
                  <img 
                    src={cnicBackPreview} 
                    alt="CNIC Back preview" 
                    className="h-full object-contain"
                  />
                ) : (
                  <span className="text-textLight">Upload Back (Image)</span>
                )}
                <input
                  type="file"
                  ref={cnicBackInputRef}
                  onChange={(e) => {
                    if (e.target.files[0]) {
                      handleFileUpload('cnicBack', e.target.files[0], true, setCnicBackPreview);
                    }
                  }}
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">JPEG, JPG or PNG (Max 5MB)</p>
            </div>
          </div>
        </div>

        {/* Professional Details Section */}
        <div className="border-b pb-6">
          <h3 className="text-lg font-semibold text-secondary mb-4">Professional Details</h3>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Medical License Number <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={formData.licenseNumber}
                onChange={(e) => handleInputChange('licenseNumber', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Years of Experience <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                required
                value={formData.experience}
                onChange={(e) => handleInputChange('experience', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
              />
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Speciality <span className="text-red-500">*</span>
            </label>
            <div className="grid grid-cols-2 gap-3 mt-2">
              {specialties.map(spec => (
                <label key={spec} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.speciality.includes(spec)}
                    onChange={() => handleSpecialityChange(spec)}
                    className="w-4 h-4 text-secondary focus:ring-secondary border-gray-300 rounded"
                  />
                  <span className="text-sm text-gray-700">{spec}</span>
                </label>
              ))}
            </div>
            {formData.speciality.length > 0 && (
              <p className="text-sm text-gray-600 mt-2">
                Selected: {formData.speciality.join(', ')}
              </p>
            )}
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Medical License Document <span className="text-red-500">*</span>
            </label>
            <div 
              className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer h-32 flex flex-col items-center justify-center"
              onClick={() => triggerFileInput(licenseInputRef)}
            >
              {licenseDocName ? (
                <div className="flex flex-col items-center justify-center">
                  <svg className="w-8 h-8 text-blue-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="text-sm text-gray-700 truncate max-w-xs">{licenseDocName}</span>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center">
                  <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <span className="text-textLight">Upload License Document (PDF)</span>
                </div>
              )}
              <input
                type="file"
                ref={licenseInputRef}
                onChange={(e) => {
                  if (e.target.files[0]) {
                    handleFileUpload('licenseDocument', e.target.files[0], false, null, setLicenseDocName);
                  }
                }}
                accept=".pdf"
                className="hidden"
              />
            </div>
            <p className="text-xs text-gray-500 mt-2">PDF only (Max 10MB)</p>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Degree Certificate <span className="text-red-500">*</span>
            </label>
            <div 
              className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer h-32 flex flex-col items-center justify-center"
              onClick={() => triggerFileInput(degreeInputRef)}
            >
              {degreeCertName ? (
                <div className="flex flex-col items-center justify-center">
                  <svg className="w-8 h-8 text-blue-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="text-sm text-gray-700 truncate max-w-xs">{degreeCertName}</span>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center">
                  <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <span className="text-textLight">Upload Degree Certificate (PDF)</span>
                </div>
              )}
              <input
                type="file"
                ref={degreeInputRef}
                onChange={(e) => {
                  if (e.target.files[0]) {
                    handleFileUpload('degreeCertificate', e.target.files[0], false, null, setDegreeCertName);
                  }
                }}
                accept=".pdf"
                className="hidden"
              />
            </div>
            <p className="text-xs text-gray-500 mt-2">PDF only (Max 10MB)</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Short Bio <span className="text-red-500">*</span> (Min. 50 characters)
            </label>
            <textarea
              required
              value={formData.bio}
              onChange={(e) => handleInputChange('bio', e.target.value)}
              rows={3}
              minLength={50}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
            />
            <p className="text-sm text-textLight mt-1">{formData.bio.length}/50 characters</p>
          </div>
        </div>

        {/* Service Details Section */}
        <div>
          <h3 className="text-lg font-semibold text-secondary mb-4">Service Details</h3>
          
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Consultation Fee (PKR) <span className="text-red-500">*</span>
            </label>
            <input
              type="number"
              required
              value={formData.consultationFee}
              onChange={(e) => handleInputChange('consultationFee', e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Available Time Slots
            </label>
            {availabilitySlots.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Your Availability:</h4>
                <div className="space-y-2">
                  {availabilitySlots.map((slot, index) => (
                    <div key={index} className="flex justify-between items-center bg-gray-50 p-3 rounded-lg">
                      <span className="text-sm">
                        {slot.day}: {slot.startTime} - {slot.endTime}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleRemoveAvailabilitySlot(index)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div 
              className="text-center border-2 border-dashed border-gray-300 rounded-lg p-6 cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => setShowAvailabilityModal(true)}
            >
              <button
                type="button"
                className="text-secondary hover:text-blue-700 font-medium"
              >
                {availabilitySlots.length > 0 ? 'Add More Time Slots' : 'Set Availability Time Slots'}
              </button>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full bg-secondary text-white py-3 rounded-lg hover:bg-blue-700 transition-colors font-semibold mt-6"
        >
          Complete Profile
        </button>
      </form>

      {/* Availability Modal */}
      {showAvailabilityModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-secondary mb-4">Add Availability Slot</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Day</label>
                <div className="relative">
                  <select 
                    value={newSlot.day}
                    onChange={(e) => setNewSlot({...newSlot, day: e.target.value})}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                  >
                    <option value="">Select Day</option>
                    {daysOfWeek.map(day => (
                      <option key={day} value={day}>{day}</option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                    <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </div>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Start Time</label>
                <div className="grid grid-cols-3 gap-2">
                  <div className="relative">
                    <select 
                      value={newSlot.startHour}
                      onChange={(e) => setNewSlot({...newSlot, startHour: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                    >
                      <option value="">Hour</option>
                      {Array.from({ length: 12 }, (_, i) => i + 1).map(hour => (
                        <option key={hour} value={hour}>{hour}</option>
                      ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <select 
                      value={newSlot.startMinute}
                      onChange={(e) => setNewSlot({...newSlot, startMinute: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                    >
                      <option value="">Min</option>
                      <option value="00">00</option>
                      <option value="15">15</option>
                      <option value="30">30</option>
                      <option value="45">45</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <select 
                      value={newSlot.startPeriod}
                      onChange={(e) => setNewSlot({...newSlot, startPeriod: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                    >
                      <option value="">AM/PM</option>
                      <option value="AM">AM</option>
                      <option value="PM">PM</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">End Time</label>
                <div className="grid grid-cols-3 gap-2">
                  <div className="relative">
                    <select 
                      value={newSlot.endHour}
                      onChange={(e) => setNewSlot({...newSlot, endHour: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                    >
                      <option value="">Hour</option>
                      {Array.from({ length: 12 }, (_, i) => i + 1).map(hour => (
                        <option key={hour} value={hour}>{hour}</option>
                      ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <select 
                      value={newSlot.endMinute}
                      onChange={(e) => setNewSlot({...newSlot, endMinute: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                    >
                      <option value="">Min</option>
                      <option value="00">00</option>
                      <option value="15">15</option>
                      <option value="30">30</option>
                      <option value="45">45</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <select 
                      value={newSlot.endPeriod}
                      onChange={(e) => setNewSlot({...newSlot, endPeriod: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary appearance-none"
                    >
                      <option value="">AM/PM</option>
                      <option value="AM">AM</option>
                      <option value="PM">PM</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                type="button"
                onClick={() => setShowAvailabilityModal(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleAddAvailabilitySlot}
                className="px-4 py-2 bg-secondary text-white rounded-lg hover:bg-blue-700"
              >
                Add Slot
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}