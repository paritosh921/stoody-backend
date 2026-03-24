// ---------------------------------------------------------------------------
// ChatPanel — embedded chat thread for objection discussions.
// Calls svc-chat directly (different base URL from teacher-bff).
// ---------------------------------------------------------------------------

import { useState, useRef, useEffect, type FormEvent } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getChatThread, sendChatMessage, type ChatMessage } from '@/api/teacher-api';

interface Props {
  examId: string;
  studentId: string;
}

export function ChatPanel({ examId, studentId }: Props) {
  const qc = useQueryClient();
  const bottomRef = useRef<HTMLDivElement>(null);
  const [message, setMessage] = useState('');

  const { data: messages, isLoading } = useQuery({
    queryKey: ['chat', examId, studentId],
    queryFn: () => getChatThread(examId, studentId),
    select: (res) => res.data?.items ?? [],
    refetchInterval: 5000, // poll every 5s for new messages
  });

  const send = useMutation({
    mutationFn: (content: string) => sendChatMessage(examId, studentId, content),
    onSuccess: () => {
      setMessage('');
      qc.invalidateQueries({ queryKey: ['chat', examId, studentId] });
    },
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!message.trim()) return;
    send.mutate(message.trim());
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="border-b border-gray-200 px-4 py-3">
        <h3 className="text-sm font-medium text-gray-900">Discussion Thread</h3>
      </div>

      <div className="h-64 overflow-y-auto px-4 py-3 space-y-3">
        {isLoading && <p className="text-sm text-gray-400">Loading messages...</p>}

        {messages?.map((msg: ChatMessage) => (
          <MessageBubble key={msg.message_id} message={msg} />
        ))}

        {messages?.length === 0 && !isLoading && (
          <p className="text-sm text-gray-400">No messages yet.</p>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-gray-200 p-3">
        <input type="text" value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message..."
          className="flex-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm
                     focus:border-brand-500 focus:outline-none focus:ring-1
                     focus:ring-brand-500" />
        <button type="submit" disabled={!message.trim() || send.isPending}
          className="rounded-md bg-brand-600 px-4 py-1.5 text-sm font-medium text-white
                     hover:bg-brand-700 disabled:opacity-50">
          Send
        </button>
      </form>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  // Simple heuristic: messages from the current teacher are right-aligned.
  // In production this would compare against the current user's ID.
  const isTeacher = message.sender_id.startsWith('teacher');
  return (
    <div className={`flex ${isTeacher ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[75%] rounded-lg px-3 py-2 text-sm ${
        isTeacher
          ? 'bg-brand-600 text-white'
          : 'bg-gray-100 text-gray-900'
      }`}>
        <p>{message.content}</p>
        <p className={`mt-1 text-[10px] ${isTeacher ? 'text-brand-200' : 'text-gray-400'}`}>
          {new Date(message.sent_at).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}
