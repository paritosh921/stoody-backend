import { useState, useRef, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchChat, sendMessage } from "@/api/student-api";
import { useAuth } from "@/hooks/useAuth";
import type { Message } from "@/types/api";
import clsx from "clsx";

function MessageBubble({
  msg,
  isMine,
}: {
  msg: Message;
  isMine: boolean;
}) {
  const time = new Date(msg.sent_at).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className={clsx("flex", isMine ? "justify-end" : "justify-start")}>
      <div
        className={clsx(
          "max-w-[75%] rounded-lg px-4 py-2 text-sm",
          isMine
            ? "bg-primary-600 text-white"
            : "bg-gray-200 text-gray-900",
        )}
      >
        <p>{msg.content}</p>
        {msg.attachment_uri && (
          <a
            href={msg.attachment_uri}
            target="_blank"
            rel="noopener noreferrer"
            className={clsx(
              "mt-1 block text-xs underline",
              isMine ? "text-blue-200" : "text-primary-600",
            )}
          >
            Attachment
          </a>
        )}
        <p
          className={clsx(
            "mt-1 text-right text-[10px]",
            isMine ? "text-blue-200" : "text-gray-400",
          )}
        >
          {time}
        </p>
      </div>
    </div>
  );
}

export default function ChatPage() {
  const { examId, teacherId } = useParams<{
    examId: string;
    teacherId: string;
  }>();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const bottomRef = useRef<HTMLDivElement>(null);
  const [text, setText] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["chat", examId, teacherId],
    queryFn: () => fetchChat(examId!, teacherId!),
    enabled: !!examId && !!teacherId,
    refetchInterval: 5000,
  });

  const mutation = useMutation({
    mutationFn: (content: string) =>
      sendMessage(examId!, teacherId!, { content }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat", examId, teacherId] });
      setText("");
    },
  });

  const messages = data?.items ?? [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  function handleSend() {
    const trimmed = text.trim();
    if (!trimmed || mutation.isPending) return;
    mutation.mutate(trimmed);
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 pb-4">
        <Link
          to={`/scores/${examId}`}
          className="text-sm text-primary-600 hover:text-primary-800"
        >
          &larr; Scores
        </Link>
        <h1 className="text-xl font-bold text-gray-900">Chat</h1>
      </div>

      {/* Messages */}
      <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border border-gray-200 bg-white p-4">
        {isLoading && (
          <p className="text-sm text-gray-400">Loading messages...</p>
        )}
        {messages.map((msg) => (
          <MessageBubble
            key={msg.message_id}
            msg={msg}
            isMine={msg.sender_id === user?.user_id}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="mt-3 flex gap-2">
        <input
          type="text"
          className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
          placeholder="Type a message..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || mutation.isPending}
          className="rounded-lg bg-primary-600 px-5 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
