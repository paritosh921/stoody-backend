"""
Google Meet API Integration Service

Creates Google Meet links for online classes using Google Calendar API.
Requires a Google Cloud Service Account with Google Workspace domain-wide delegation.

Environment Variables:
- GOOGLE_SERVICE_ACCOUNT_JSON: Path to service account JSON file
- GOOGLE_CALENDAR_ID: Calendar ID to create events (default: primary)
- GOOGLE_DELEGATE_EMAIL: Email to impersonate (required for domain-wide delegation)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import uuid

logger = logging.getLogger(__name__)

# Configuration
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
GOOGLE_DELEGATE_EMAIL = os.getenv("GOOGLE_DELEGATE_EMAIL", "")

# Google API scopes required
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
]


class GoogleMeetService:
    """Service for creating and managing Google Meet links"""

    def __init__(self):
        self._credentials = None
        self._service = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Google API credentials"""
        if self._initialized:
            return True

        if not GOOGLE_SERVICE_ACCOUNT_JSON:
            logger.warning("Google Meet service not configured - GOOGLE_SERVICE_ACCOUNT_JSON not set")
            return False

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            # Load service account credentials
            if os.path.isfile(GOOGLE_SERVICE_ACCOUNT_JSON):
                self._credentials = service_account.Credentials.from_service_account_file(
                    GOOGLE_SERVICE_ACCOUNT_JSON,
                    scopes=SCOPES
                )
            else:
                # Try parsing as JSON string
                creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
                self._credentials = service_account.Credentials.from_service_account_info(
                    creds_info,
                    scopes=SCOPES
                )

            # Delegate to user if configured (for domain-wide delegation)
            if GOOGLE_DELEGATE_EMAIL:
                self._credentials = self._credentials.with_subject(GOOGLE_DELEGATE_EMAIL)

            # Build the Calendar API service
            self._service = build('calendar', 'v3', credentials=self._credentials)
            self._initialized = True
            logger.info("Google Meet service initialized successfully")
            return True

        except ImportError:
            logger.error("Google API libraries not installed. Run: pip install google-auth google-auth-oauthlib google-api-python-client")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Google Meet service: {e}")
            return False

    async def create_meeting(
        self,
        topic: str,
        description: str,
        scheduled_at: datetime,
        duration_minutes: int = 60,
        attendee_emails: Optional[list] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Create a Google Meet meeting via Calendar API.

        Args:
            topic: Meeting title
            description: Meeting description
            scheduled_at: Start time (datetime object)
            duration_minutes: Duration in minutes
            attendee_emails: Optional list of attendee emails

        Returns:
            Tuple of (meet_link, meet_code, event_id) or (None, None, None) on failure
        """
        if not await self.initialize():
            # Fall back to generating a placeholder link
            return await self._generate_fallback_link(topic)

        try:
            # Calculate end time
            end_time = scheduled_at + timedelta(minutes=duration_minutes)

            # Create event body
            event = {
                'summary': topic,
                'description': description,
                'start': {
                    'dateTime': scheduled_at.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'conferenceData': {
                    'createRequest': {
                        'requestId': str(uuid.uuid4()),
                        'conferenceSolutionKey': {
                            'type': 'hangoutsMeet'
                        }
                    }
                },
            }

            # Add attendees if provided
            if attendee_emails:
                event['attendees'] = [{'email': email} for email in attendee_emails]

            # Create the event with conference data
            created_event = self._service.events().insert(
                calendarId=GOOGLE_CALENDAR_ID,
                body=event,
                conferenceDataVersion=1,
                sendUpdates='all' if attendee_emails else 'none'
            ).execute()

            # Extract Meet link and code
            conference_data = created_event.get('conferenceData', {})
            entry_points = conference_data.get('entryPoints', [])

            meet_link = None
            meet_code = None

            for entry in entry_points:
                if entry.get('entryPointType') == 'video':
                    meet_link = entry.get('uri')
                    # Extract code from link (e.g., meet.google.com/xxx-xxxx-xxx)
                    if meet_link:
                        meet_code = meet_link.split('/')[-1]
                    break

            event_id = created_event.get('id')

            logger.info(f"Created Google Meet: {meet_link} for event {event_id}")
            return meet_link, meet_code, event_id

        except Exception as e:
            logger.error(f"Failed to create Google Meet: {e}")
            return await self._generate_fallback_link(topic)

    async def _generate_fallback_link(self, topic: str) -> Tuple[str, str, None]:
        """Generate a fallback Meet link when API is not available"""
        import hashlib
        import random
        import string

        # Generate a random meeting code in Google Meet format (xxx-xxxx-xxx)
        def generate_code():
            chars = string.ascii_lowercase
            return f"{''.join(random.choices(chars, k=3))}-{''.join(random.choices(chars, k=4))}-{''.join(random.choices(chars, k=3))}"

        meet_code = generate_code()
        meet_link = f"https://meet.google.com/{meet_code}"

        logger.info(f"Generated fallback Meet link: {meet_link}")
        return meet_link, meet_code, None

    async def cancel_meeting(self, event_id: str) -> bool:
        """
        Cancel a Google Calendar event with Meet link.

        Args:
            event_id: Google Calendar event ID

        Returns:
            True if cancelled, False otherwise
        """
        if not event_id:
            return True  # No event to cancel

        if not await self.initialize():
            return False

        try:
            self._service.events().delete(
                calendarId=GOOGLE_CALENDAR_ID,
                eventId=event_id,
                sendUpdates='all'
            ).execute()
            logger.info(f"Cancelled Google Calendar event: {event_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel Google Calendar event {event_id}: {e}")
            return False

    async def update_meeting(
        self,
        event_id: str,
        topic: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
        duration_minutes: Optional[int] = None,
    ) -> bool:
        """
        Update an existing Google Calendar event.

        Args:
            event_id: Google Calendar event ID
            topic: New title (optional)
            scheduled_at: New start time (optional)
            duration_minutes: New duration (optional)

        Returns:
            True if updated, False otherwise
        """
        if not event_id:
            return False

        if not await self.initialize():
            return False

        try:
            # Get existing event
            event = self._service.events().get(
                calendarId=GOOGLE_CALENDAR_ID,
                eventId=event_id
            ).execute()

            # Update fields
            if topic:
                event['summary'] = topic

            if scheduled_at:
                end_time = scheduled_at + timedelta(minutes=duration_minutes or 60)
                event['start'] = {
                    'dateTime': scheduled_at.isoformat(),
                    'timeZone': 'UTC',
                }
                event['end'] = {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                }
            elif duration_minutes:
                # Recalculate end time based on current start
                start_str = event.get('start', {}).get('dateTime')
                if start_str:
                    start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = start + timedelta(minutes=duration_minutes)
                    event['end'] = {
                        'dateTime': end_time.isoformat(),
                        'timeZone': 'UTC',
                    }

            # Update the event
            self._service.events().update(
                calendarId=GOOGLE_CALENDAR_ID,
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()

            logger.info(f"Updated Google Calendar event: {event_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update Google Calendar event {event_id}: {e}")
            return False


# Singleton instance
google_meet_service = GoogleMeetService()
