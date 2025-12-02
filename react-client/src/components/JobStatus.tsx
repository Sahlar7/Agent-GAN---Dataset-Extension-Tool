// src/components/JobStatus.tsx - Complete, Safe Component

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import BackgroundPattern from './BackgroundPattern';

// -------------------------------------------------------------
// Type Definitions
// -------------------------------------------------------------
interface Message {
  role: 'user' | 'agent';
  content: string;
  timestamp: string;
}

// NOTE: Added 'cancelled' to the status union type
type JobStatusType =
  | 'queued'
  | 'analyzing'
  | 'awaiting_input'
  | 'submitted'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

interface Job {
  job_id: string;
  status: JobStatusType;
  messages?: Message[];
  slurm_job_id?: string;
  result_url?: string;
  // Added optional field for cancellation details
  cancellation_reason?: string;
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
  // Added shadow color for finished states
  shadowColor: string;
}

// -------------------------------------------------------------
// Status Configuration
// -------------------------------------------------------------
const getStatusConfig = (status: JobStatusType | 'default'): StatusConfig => {
  const configs: Record<JobStatusType | 'default', StatusConfig> = {
    completed: {
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      icon: (
        <svg
          className="w-16 h-16 text-green-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      shadowColor: 'shadow-green-500/50',
    },
    failed: {
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      icon: (
        <svg
          className="w-16 h-16 text-red-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      shadowColor: 'shadow-red-500/50',
    },
    cancelled: {
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      icon: (
        <svg
          className="w-16 h-16 text-yellow-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      shadowColor: 'shadow-yellow-500/50',
    },
    awaiting_input: {
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      icon: (
        <svg
          className="w-16 h-16 text-blue-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
          />
        </svg>
      ),
      shadowColor: 'shadow-none',
    },
    queued: {
      color: 'text-gray-700',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: null,
      shadowColor: 'shadow-none',
    },
    analyzing: {
      color: 'text-gray-700',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: null,
      shadowColor: 'shadow-none',
    },
    submitted: {
      color: 'text-gray-700',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: null,
      shadowColor: 'shadow-none',
    },
    running: {
      color: 'text-gray-700',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: null,
      shadowColor: 'shadow-none',
    },
    default: {
      color: 'text-gray-700',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: null,
      shadowColor: 'shadow-none',
    },
  };

  return configs[status] || configs.default;
};

// -------------------------------------------------------------
// Helper Component: Confetti (Simple CSS-based)
// -------------------------------------------------------------
const Confetti: React.FC = () => (
  <>
    <div className="confetti-container absolute inset-0 overflow-hidden pointer-events-none">
      {Array.from({ length: 50 }).map((_, i) => (
        <div
          key={i}
          className="confetto absolute w-3 h-3 rounded-full opacity-0"
          style={{
            left: `${Math.random() * 100}%`,
            backgroundColor: `hsl(${Math.random() * 360}, 100%, 70%)`,
            animationDelay: `${Math.random() * 2}s`,
            animationDuration: `${Math.random() * 3 + 2}s`,
          }}
        />
      ))}
    </div>
    <style>{`
      @keyframes fall {
        0% { transform: translateY(-100px) rotate(0deg); opacity: 1; }
        100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
      }
      .confetto {
        animation: fall linear infinite;
        top: 0;
      }
    `}</style>
  </>
);

// -------------------------------------------------------------
// Helper Component: Status Content Block (For In-Progress)
// -------------------------------------------------------------
const StatusContentBlock: React.FC<{ job: Job }> = ({ job }) => {
  const isProcessing = ['queued', 'analyzing', 'submitted', 'running'].includes(
    job.status
  );

  const PulseLoader = () => (
    <div className="relative w-16 h-16 flex items-center justify-center">
      <div className="absolute w-full h-full border-4 border-gray-300 rounded-full animate-ping-slow" />
      <svg
        className="w-8 h-8 text-gray-700"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M4 4v5h.582m15.836 2.5a8.001 8.001 0 00-15.836 0m12.663 1.5a2.002 2.002 0 100-4 2.002 2.002 0 000 4z"
        />
      </svg>
      <div className="absolute w-2 h-2 bg-gray-500 rounded-full" />
    </div>
  );

  const getProcessingMessage = (status: JobStatusType) => {
    switch (status) {
      case 'queued':
        return {
          title: 'Job Queued',
          message:
            'Your job is in the queue, awaiting resource allocation and processing.',
        };
      case 'analyzing':
        return {
          title: 'Analyzing Input',
          message:
            'The Datagent is reviewing your prompt and dataset to determine the augmentation strategy.',
        };
      case 'submitted':
        return {
          title: 'SLURM Job Submitted',
          message:
            'The augmentation task has been submitted to the HPC cluster and is awaiting execution.',
        };
      case 'running':
        return {
          title: 'Processing Data',
          message:
            'The GAN-based augmentation is actively generating synthetic data points.',
        };
      default:
        return {
          title: 'Processing...',
          message: 'Please wait while the job is processed.',
        };
    }
  };

  const content = getProcessingMessage(job.status);

  return (
    <div className="text-center py-12">
      <div className="flex justify-center mb-6">
        {isProcessing ? <PulseLoader /> : null}
      </div>
      <p className="text-xl font-bold text-gray-700 mb-2">{content.title}</p>
      <p className="text-gray-500 max-w-sm mx-auto text-sm">
        {content.message}
      </p>
    </div>
  );
};

// -------------------------------------------------------------
// New Component: Job Finished View (Completed, Failed, Cancelled)
// -------------------------------------------------------------
interface JobFinishedViewProps {
  job: Job;
  config: StatusConfig;
  statusLabel: string;
  onReset: () => void;
}

const JobFinishedView: React.FC<JobFinishedViewProps> = ({
  job,
  config,
  statusLabel,
  onReset,
}) => {
  const isSuccess = job.status === 'completed';
  const isCancelled = job.status === 'cancelled';
  const isFailed = job.status === 'failed';
  const API_URL = "http://localhost:8100"
  const submitDownloadRequest = async () => {
    if (!job) return;

    try {
      const response = await axios.get(
        `${API_URL}/api/jobs/${job.job_id}/result`,
        {
          responseType: 'blob',       // 👈 get binary
          withCredentials: true,      // 👈 if you rely on cookies
        }
      );

      // Try to extract filename from Content-Disposition
      const disposition = response.headers['content-disposition'];
      let filename = 'result.zip';
      if (disposition) {
        const match = disposition.match(/filename="?(.+?)"?$/i);
        if (match && match[1]) {
          filename = match[1];
        }
      }

      // Create a blob URL and trigger a download
      const blob = new Blob([response.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename); // hint to browser
      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      // optionally set some UI error state here
    }
  };


  return (
    <div
      className={`relative ${config.bgColor} border-2 ${config.borderColor} rounded-2xl p-10 shadow-2xl ${config.shadowColor} transition-shadow duration-500`}
    >
      {isSuccess && <Confetti />}

      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">{config.icon}</div>
        <h2 className={`text-4xl font-extrabold ${config.color} mb-2`}>
          {statusLabel}
        </h2>

        {isSuccess && (
          <p className="text-gray-600">
            The data augmentation job finished successfully. Your results are
            ready to download.
          </p>
        )}

        {isFailed && (
          <p className="text-gray-600">
            The job encountered an issue and was unable to complete. Please
            check the details below.
          </p>
        )}

        {isCancelled && (
          <p className="text-gray-600">
            The job was cancelled.{' '}
            {job.cancellation_reason
              ? `Reason: ${job.cancellation_reason}`
              : 'No specific reason provided.'}
          </p>
        )}
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-6 mb-8 space-y-4">
        <h3 className="text-lg font-semibold text-gray-800 border-b pb-2 mb-4">
          Job Details
        </h3>

        {/* Job ID */}
        <div className="flex justify-between items-center">
          <p className="text-sm text-gray-500">Job UUID</p>
          <p className="font-mono text-gray-700 text-sm">{job.job_id}</p>
        </div>

        {/* SLURM ID */}
        {job.slurm_job_id && (
          <div className="flex justify-between items-center border-t pt-2">
            <p className="text-sm text-gray-500">SLURM ID</p>
            <p className="font-mono text-gray-700 text-sm">
              {job.slurm_job_id}
            </p>
          </div>
        )}

        {/* Status */}
        <div className="flex justify-between items-center border-t pt-2">
          <p className="text-sm text-gray-500">Final Status</p>
          <p className={`font-bold capitalize text-sm ${config.color}`}>
            {statusLabel}
          </p>
        </div>
      </div>

      {/* Artifacts Download Section (Only for Completed) */}
      {isSuccess ? (
        <div className="text-center mb-8">
          <button
            onClick={submitDownloadRequest}
            className="inline-flex items-center justify-center px-8 py-4 border border-transparent 
              text-lg font-semibold rounded-xl text-white bg-blue-600 
              hover:bg-blue-700 transition-all duration-300 shadow-lg hover:shadow-xl
              active:scale-[0.98]"
          >
            <svg
              className="w-6 h-6 mr-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            Download Artifacts
          </button>
        </div>
      ) : (
        <div className="text-center mb-8 text-gray-500 text-sm">
          {isSuccess
            ? 'Artifact URL is missing.'
            : 'No artifacts available for download.'}
        </div>
      )}

      {/* Reset Button */}
      <div className="text-center pt-4 border-t border-gray-200">
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

// -------------------------------------------------------------
// JobContent Component for Clean Separation
// -------------------------------------------------------------
interface JobContentProps {
  job: Job;
  jobId: string;
  getStatusConfig: (status: JobStatusType | 'default') => StatusConfig;
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
  submitting,
}) => {
  const statusValue: JobStatusType | 'default' = job.status ?? 'default';
  const config = getStatusConfig(statusValue);
  const isInteractive = job.status === 'awaiting_input';

  const isFinished = ['completed', 'failed', 'cancelled'].includes(job.status);
  console.log(job.status);

  const capitalizeFirstLetter = (val:any) =>  {
    return String(val).charAt(0).toUpperCase() + String(val).slice(1);
  }

  const messages = job.messages ?? [];

  const statusLabel = capitalizeFirstLetter(job.status) ? job.status.replace('_', ' ') : 'Unknown status';

  if (isFinished) {
    return (
      <JobFinishedView
        job={job}
        config={config}
        statusLabel={statusLabel}
        onReset={onReset}
      />
    );
  }

  return (
    <div className={`${config.bgColor} border-2 ${config.borderColor} rounded-2xl p-8`}>
      {/* Status Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">
            Job ID
          </p>
          <p className="text-sm font-mono text-gray-700">{jobId}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">
            Status
          </p>
          <p className={`text-lg font-bold capitalize ${config.color}`}>
            {statusLabel}
          </p>
        </div>
      </div>

      {/* SLURM Job ID if submitted */}
      {job.slurm_job_id && (
        <div className="mb-6 bg-white/70 border border-gray-300 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">
            SLURM Job ID
          </p>
          <p className="text-base font-mono text-gray-800">{job.slurm_job_id}</p>
        </div>
      )}

      {/* Status Indicator Block */}
      {!isInteractive && messages.length === 0 && <StatusContentBlock job={job} />}

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
              className="w-full px-6 py-5 pr-20
                bg-white border-2 border-gray-300 rounded-xl
                text-gray-900 placeholder-gray-400
                focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-600/10
                resize-none transition-all duration-300
                disabled:opacity-50 disabled:cursor-not-allowed"
            />

            <button
              onClick={handleSendMessage}
              disabled={submitting || !userInput.trim()}
              className="absolute bottom-5 right-5
                w-12 h-12 rounded-xl
                bg-blue-600 hover:bg-blue-700
                flex items-center justify-center
                transition-all duration-300
                disabled:opacity-40 disabled:cursor-not-allowed
                hover:shadow-lg active:scale-95"
            >
              {submitting ? (
                <svg
                  className="animate-spin w-5 h-5 text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              ) : (
                <svg
                  className="w-5 h-5 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2.5}
                    d="M5 10l7-7m0 0l7 7m-7-7v18"
                  />
                </svg>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send, Shift+Enter for new line
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

// -------------------------------------------------------------
// JobStatus Component (Container)
// -------------------------------------------------------------
export default function JobStatus({ jobId, onReset }: JobStatusProps) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [userInput, setUserInput] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [pollingStopped, setPollingStopped] = useState(false);

  const API_URL = 'http://localhost:8100';
  const terminalStatuses: JobStatusType[] = ['completed', 'failed', 'cancelled'];

  // Refs to keep latest values inside async callbacks (avoid stale closures)
  const jobRef = useRef<Job | null>(null);
  const pollingStoppedRef = useRef(false);



  useEffect(() => {
    jobRef.current = job;
  }, [job]);

  useEffect(() => {
    pollingStoppedRef.current = pollingStopped;
  }, [pollingStopped]);

  // Reset local state when jobId changes
  useEffect(() => {
    setJob(null);
    setError(null);
    setUserInput('');
    setSubmitting(false);
    setPollingStopped(false);
  }, [jobId]);

  const pollJob = async () => {
    if (pollingStoppedRef.current) return true;

    try {
      const response = await axios.get<Job>(`${API_URL}/api/jobs/${jobId}`);
      console.log(response.data);

      setJob(response.data);
      setError(null);

      const done = terminalStatuses.includes(response.data.status);
      if (done) {
        setPollingStopped(true);
      }
      return done;
    } catch (err: any) {
      console.error('Polling Error:', err);

      const lastJob = jobRef.current;
      const status = err?.response?.status;

      // If we've *ever* seen this job (lastJob != null) and now it's 404,
      // assume the backend cleaned it up after finishing. Stop polling silently.
      if (status === 404 && lastJob) {
        setPollingStopped(true);
        setError(null);
        return true;
      }

      // Otherwise it's a genuine error (invalid ID / never existed / API down)
      setError(
        status === 404
          ? 'Job not found. It may have been deleted or the ID is invalid.'
          : 'Failed to fetch job status. The API may be down.'
      );
      setPollingStopped(true);
      return true;
    }
  };

  useEffect(() => {
    let isCancelled = false;
    let delay = 3000;
    const maxDelay = 60000;

    const schedulePoll = async () => {
      if (isCancelled || pollingStoppedRef.current) return;

      const done = await pollJob();

      if (done || isCancelled || pollingStoppedRef.current) return;

      delay = Math.min(delay * 1.6, maxDelay);

      setTimeout(schedulePoll, delay);
    };

    schedulePoll();

    return () => {
      isCancelled = true;
    };
  }, [jobId]); // restart polling when jobId changes

  const handleSendMessage = async () => {
    if (!userInput.trim() || !job) return;

    setSubmitting(true);
    setError(null);

    try {
      await axios.post(`${API_URL}/api/jobs/${jobId}/respond`, {
        message: userInput,
        conversation_history: job.messages ?? [],
      });

      const newMessage: Message = {
        role: 'user',
        content: userInput,
        timestamp: new Date().toISOString(),
      };

      setJob({
        ...job,
        messages: [...(job.messages ?? []), newMessage],
        status: 'analyzing',
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
            {/* Error Message Display */}
            {error && (
              <div className="bg-red-50 border-2 border-red-200 rounded-2xl px-6 py-4 mb-8">
                <p className="text-red-700 text-sm font-medium">{error}</p>
              </div>
            )}

            {!job ? (
              <div className="text-center py-16">
                <svg
                  className="animate-spin w-16 h-16 mx-auto mb-6 text-gray-400"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
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
