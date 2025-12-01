// src/components/JobStatus.tsx - Complete, Safe Component

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import BackgroundPattern from './BackgroundPattern';

interface Message {
  role: 'user' | 'agent';
  content: string;
  timestamp: string;
}

interface Job {
  job_id: string;
  status: 'queued' | 'analyzing' | 'awaiting_input' | 'submitted' | 'running' | 'completed' | 'failed';
  messages?: Message[];
  slurm_job_id?: string;
  result_url?: string;
}

interface JobStatusProps {
  jobId: string;
  onReset: () => void;
}

interface StatusConfig {
  color: string;
  bgColor: string;
  borderColor: string;
  icon: any | null;
}

const getStatusConfig = (status: string): StatusConfig => {
  const configs: Record<string, StatusConfig> = {
    completed: {
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      icon: (
        <svg className="w-16 h-16 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    failed: {
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      icon: (
        <svg className="w-16 h-16 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    awaiting_input: {
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      icon: (
        <svg className="w-16 h-16 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
      )
    },
    default: {
      color: 'text-gray-700',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: null
    }
  };

  return configs[status] || configs.default;
};

// -------------------------------------------------------------
// Helper Component: Status Content Block
// -------------------------------------------------------------
const StatusContentBlock: React.FC<{ job: Job }> = ({ job }) => {
  const isProcessing = ['queued', 'analyzing', 'submitted', 'running'].includes(job.status);

  // Custom Pulse Animation for Processing States
  const PulseLoader = () => (
    <div className="relative w-16 h-16 flex items-center justify-center">
      {/* Outer Pulse */}
      <div className="absolute w-full h-full border-4 border-gray-300 rounded-full animate-ping-slow" />
      {/* Inner Icon */}
      <svg className="w-8 h-8 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.836 2.5a8.001 8.001 0 00-15.836 0m12.663 1.5a2.002 2.002 0 100-4 2.002 2.002 0 000 4z" />
      </svg>
      {/* Small Dot */}
      <div className="absolute w-2 h-2 bg-gray-500 rounded-full" />
    </div>
  );

  const getProcessingMessage = (status: string) => {
    switch (status) {
      case 'queued':
        return { title: 'Job Queued', message: 'Your job is in the queue, awaiting resource allocation and processing.' };
      case 'analyzing':
        return { title: 'Analyzing Input', message: 'The Datagent is reviewing your prompt and dataset to determine the augmentation strategy.' };
      case 'submitted':
        return { title: 'SLURM Job Submitted', message: 'The augmentation task has been submitted to the HPC cluster and is awaiting execution.' };
      case 'running':
        return { title: 'Processing Data', message: 'The GAN-based augmentation is actively generating synthetic data points.' };
      default:
        return { title: 'Processing...', message: 'Please wait while the job is processed.' };
    }
  };

  const content = getProcessingMessage(job.status);

  return (
    <div className="text-center py-12">
      <div className="flex justify-center mb-6">
        {isProcessing ? <PulseLoader /> : null}
      </div>
      <p className="text-xl font-bold text-gray-700 mb-2">
        {content.title}
      </p>
      <p className="text-gray-500 max-w-sm mx-auto text-sm">
        {content.message}
      </p>
    </div>
  );
};

// -------------------------------------------------------------
// JobStatus Component
// -------------------------------------------------------------
export default function JobStatus({ jobId, onReset }: JobStatusProps) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [userInput, setUserInput] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const API_URL = 'http://localhost:8100';

  const pollJob = async () => {
      try {
        const response = await axios.get<Job>(`${API_URL}/api/jobs/${jobId}`);
        console.log(response.data)
        setJob(response.data);

        // Stop polling if complete or failed
        return response.data.status === 'completed' || response.data.status === 'failed';
      } catch (err) {
        console.error(err)
        // Stop polling on error to prevent continuous failing requests
        setError('Failed to fetch job status');
        return true;
      }
    };

  useEffect(() => {
    let isCancelled = false;
    let delay = 3000;                   // start at 3 seconds
    const maxDelay = 15 * 60 * 1000;    // cap at 15 minutes

    const schedulePoll = async () => {
      const done = await pollJob();
      if (done || isCancelled) return;

      // Soft exponential backoff (gently increases)
      delay = Math.min(delay * 1.6, maxDelay);

      setTimeout(schedulePoll, delay);
    };

    // Kick off immediately:
    schedulePoll();

    return () => {
      isCancelled = true;
    };
  }, [jobId]);


  const handleSendMessage = async () => {
    if (!userInput.trim() || !job) return;

    setSubmitting(true);
    setError(null);

    try {
      // 1. Send user message to the API
      await axios.post(`${API_URL}/api/jobs/${jobId}/respond`, {
        message: userInput,
        conversation_history: job.messages ?? []
      });

      // 2. Optimistically update local state to show the user's message immediately
      const newMessage: Message = {
        role: 'user',
        content: userInput,
        timestamp: new Date().toISOString()
      };

      setJob({
        ...job,
        messages: [...(job.messages ?? []), newMessage],
        // Transition status to 'analyzing' to show the agent is processing the input
        status: 'analyzing'
      });

      setUserInput('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to send message');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      <BackgroundPattern />

      {/* Tailwind CSS keyframe for the pulse animation */}
      <style>{`
        @keyframes pulse-slow {
          0%, 100% {
            transform: scale(1);
            opacity: 1;
          }
          50% {
            transform: scale(1.5);
            opacity: 0;
          }
        }
        .animate-ping-slow {
          animation: pulse-slow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
      `}</style>

      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="border-b border-gray-200 py-8 px-6">
          <h1 className="text-5xl text-center font-serif italic text-gray-900">
            Datagent
          </h1>
        </header>

        {/* Main Content */}
        <main className="flex-1 flex items-center justify-center px-6 py-16">
          <div className="w-full max-w-4xl">
            {error && (
              <div className="bg-red-50 border-2 border-red-200 rounded-2xl px-6 py-4 mb-8">
                <p className="text-red-700 text-sm font-medium">{error}</p>
              </div>
            )}

            {!job ? (
              <div className="text-center py-16">
                <svg className="animate-spin w-16 h-16 mx-auto mb-6 text-gray-400" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <p className="text-gray-600">Loading job status...</p>
              </div>
            ) : (
              <JobContent
                job={job}
                jobId={jobId}
                getStatusConfig={getStatusConfig}
                onReset={onReset}
                handleSendMessage={handleSendMessage}
                userInput={userInput}
                setUserInput={setUserInput}
                submitting={submitting}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// JobContent Component for Clean Separation
// This component encapsulates the logic that requires the 'job' object.
// -------------------------------------------------------------
interface JobContentProps {
  job: Job;
  jobId: string;
  getStatusConfig: (status: string) => StatusConfig;
  onReset: () => void;
  handleSendMessage: () => Promise<void>;
  userInput: string;
  setUserInput: React.Dispatch<React.SetStateAction<string>>;
  submitting: boolean;
}

const JobContent: React.FC<JobContentProps> = ({
  job,
  jobId,
  getStatusConfig,
  onReset,
  handleSendMessage,
  userInput,
  setUserInput,
  submitting
}) => {
  // SAFETY: use a default for config lookup
  const statusValue = job.status ?? 'default';
  const config = getStatusConfig(statusValue);
  const isInteractive = job.status === 'awaiting_input';

  // SAFETY: guard messages
  const messages = job.messages ?? [];

  // SAFETY: avoid calling .replace on undefined
  const statusLabel = job.status ? job.status.replace('_', ' ') : 'Unknown status';

  return (
    <div className={`${config.bgColor} border-2 ${config.borderColor} rounded-2xl p-8`}>

      {/* Status Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Job ID</p>
          <p className="text-sm font-mono text-gray-700">{jobId}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Status</p>
          <p className={`text-lg font-bold capitalize ${config.color}`}>
            {statusLabel}
          </p>
        </div>
      </div>

      {/* SLURM Job ID if submitted */}
      {job.slurm_job_id && (
        <div className="mb-6 bg-white/70 border border-gray-300 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">SLURM Job ID</p>
          <p className="text-base font-mono text-gray-800">{job.slurm_job_id}</p>
        </div>
      )}

      {/* Status Indicator Block (For non-interactive, non-conversation states) */}
      {!isInteractive && messages.length === 0 && (
        <StatusContentBlock job={job} />
      )}

      {/* Conversation History */}
      {messages.length > 0 && (
        <div className="mb-6 space-y-4 max-h-96 overflow-y-auto">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-xl ${
                msg.role === 'user'
                  ? 'bg-gray-100 border border-gray-200 ml-8'
                  : 'bg-white/70 border border-gray-300 mr-8'
              }`}
            >
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
                {msg.role === 'user' ? 'You' : 'Agent'}
              </p>
              <p className="text-gray-800 whitespace-pre-wrap">{msg.content}</p>
            </div>
          ))}
        </div>
      )}

      {/* Input Area (only for awaiting_input) */}
      {isInteractive && (
        <div className="mb-6">
          <div className="relative">
            <textarea
              value={userInput}
              onChange={(e) => {
                setUserInput(e.target.value);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="Type your response..."
              rows={4}
              disabled={submitting}
              className="
                w-full px-6 py-5 pr-20
                bg-white border-2 border-gray-300 rounded-xl
                text-gray-900 placeholder-gray-400
                focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-600/10
                resize-none transition-all duration-300
                disabled:opacity-50 disabled:cursor-not-allowed
              "
            />

            <button
              onClick={handleSendMessage}
              disabled={submitting || !userInput.trim()}
              className="
                absolute bottom-5 right-5
                w-12 h-12 rounded-xl
                bg-blue-600 hover:bg-blue-700
                flex items-center justify-center
                transition-all duration-300
                disabled:opacity-40 disabled:cursor-not-allowed
                hover:shadow-lg active:scale-95
              "
            >
              {submitting ? (
                <svg className="animate-spin w-5 h-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">Press Enter to send, Shift+Enter for new line</p>
        </div>
      )}

      {/* Result URL if completed */}
      {job.status === 'completed' && job.result_url && (
        <div className="mb-6 bg-white/70 border border-gray-300 rounded-xl p-6">
          <p className="text-sm text-gray-700 mb-3">
            Training complete! Your augmented dataset is available on HPC:
          </p>
          <p className="text-sm font-mono bg-gray-100 p-3 rounded border border-gray-200 break-all">
            {job.result_url}
          </p>
        </div>
      )}

      {/* Reset Button */}
      <div className="text-center pt-4">
        <button
          onClick={onReset}
          className="text-gray-600 hover:text-gray-900 font-medium transition-colors duration-300"
        >
          ← Start New Job
        </button>
      </div>
    </div>
  );
};
