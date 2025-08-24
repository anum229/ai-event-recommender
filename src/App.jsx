// src/App.jsx
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import CompleteProfile from './pages/CompleteProfile';
import LandingPage from './pages/LandingPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/complete-profile" element={<CompleteProfile />} />
          <Route path="/landing" element={<LandingPage />} />
          <Route path="/signin" element={<div className="text-center mt-10">Sign In Page - Placeholder</div>} />
          {/* Default redirect to complete profile for testing */}
          <Route path="/" element={<Navigate to="/complete-profile?role=Patient" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;