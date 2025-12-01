// src/components/UploadScreen.tsx
import { useState } from 'react';
import axios from 'axios';
import BackgroundPattern from './BackgroundPattern';

interface UploadScreenProps {
  onJobCreated: (jobId: string) => void;
}

export default function UploadScreen({ onJobCreated }: UploadScreenProps) {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL = 'http://localhost:8100';

  const handleSubmit = async () => {
    if (!file || !prompt.trim()) {
      setError('Please upload a dataset and provide a description');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('dataset', file);
    formData.append('prompt', prompt);

    try {
      const response = await axios.post(API_URL + '/api/jobs', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      onJobCreated(response.data.job_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit job');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      <BackgroundPattern />

      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="border-b border-gray-200 py-8 px-6">
          <h1 className="text-5xl text-center font-serif italic text-gray-900">
            Datagent
          </h1>
          <p className="text-center text-gray-500 mt-3 text-xs tracking-widest uppercase">
            AI-Powered Dataset Augmentation
          </p>
        </header>

        {/* Main Content */}
        <main className="flex-1 flex items-center justify-center px-6 py-16">
          <div className="w-full max-w-4xl space-y-12">
            
            {/* Hero Section */}
            <div className="text-center space-y-4">
              <h2 className="text-5xl font-bold text-gray-900 leading-tight">
                Automated Dataset <span className="text-green-600">Expansion</span>
              </h2>
              <p className="text-gray-600 text-lg max-w-2xl mx-auto leading-relaxed">
                Upload your dataset and let our AI agent handle the augmentation process with intelligent GAN-based synthesis
              </p>
            </div>

            {/* Upload Dropzone */}
            <div>
              <input
                type="file"
                accept=".zip"
                onChange={(e) => {
                  setFile(e.target.files?.[0] || null);
                  setError(null);
                }}
                className="hidden"
                id="file-upload"
                disabled={loading}
              />
              <label
                htmlFor="file-upload"
                className={`
                  flex items-center justify-center w-full h-64
                  border-2 border-dashed rounded-2xl
                  transition-all duration-300 cursor-pointer
                  ${file 
                    ? 'border-green-600 bg-green-50' 
                    : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
                  }
                  ${loading ? 'opacity-50 cursor-not-allowed' : ''}
                `}
              >
                {file ? (
                  <div className="text-center px-6">
                    <svg className="w-16 h-16 mx-auto mb-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="text-xl font-semibold text-gray-900 mb-1">{file.name}</p>
                    <p className="text-sm text-gray-500">Click to replace dataset</p>
                  </div>
                ) : (
                  <div className="text-center px-6">
                    <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-xl font-medium text-gray-700 mb-1">Upload Dataset</p>
                    <p className="text-sm text-gray-400 uppercase tracking-wide">ZIP files only</p>
                  </div>
                )}
              </label>
            </div>

            {/* Prompt Input */}
            <div className="relative">
              <textarea
                value={prompt}
                onChange={(e) => {
                  setPrompt(e.target.value);
                  setError(null);
                }}
                placeholder="Start typing to relay any important information about your data/task..."
                rows={8}
                disabled={loading}
                className="
                  w-full px-6 py-5 pr-20
                  bg-gray-50 border-2 border-gray-300 rounded-2xl
                  text-gray-900 placeholder-gray-400
                  focus:outline-none focus:border-green-600 focus:ring-4 focus:ring-green-600/10
                  resize-none transition-all duration-300
                  disabled:opacity-50 disabled:cursor-not-allowed
                "
              />
              
              {/* Submit Button */}
              <button 
                onClick={handleSubmit}
                disabled={loading || !file || !prompt.trim()}
                className="
                  absolute bottom-5 right-5
                  w-14 h-14 rounded-xl
                  bg-green-600 hover:bg-green-700
                  flex items-center justify-center
                  transition-all duration-300
                  disabled:opacity-40 disabled:cursor-not-allowed
                  hover:shadow-lg active:scale-95
                "
              >
                {loading ? (
                  <svg className="animate-spin w-6 h-6 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                )}
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border-2 border-red-200 rounded-xl px-6 py-4">
                <p className="text-red-700 text-sm font-medium">{error}</p>
              </div>
            )}

            {/* Helper Text */}
            <p className="text-center text-gray-500 text-sm">
              Provide details about your dataset structure, size, and augmentation goals
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}