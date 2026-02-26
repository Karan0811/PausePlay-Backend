from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
import razorpay
import random

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Razorpay client (using test keys)
RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID', 'rzp_test_dummy')
RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET', 'dummy_secret')
print(f"Using Razorpay Key ID: {RAZORPAY_KEY_ID}")
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# JWT Secret
JWT_SECRET = os.environ.get('JWT_SECRET', 'pause-play-secret-key-2025')
JWT_ALGORITHM = 'HS256'

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ============= MODELS =============

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    name: str
    phone: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    sport: str
    image: str
    date: str
    time: str
    location: str
    city: str
    price: int
    total_spots: int
    spots_remaining: int
    organizer: str
    venue_details: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_popular: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EventFilters(BaseModel):
    sport: Optional[str] = None
    city: Optional[str] = None
    date: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    search: Optional[str] = None

class Booking(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    event_id: str
    quantity: int
    total_amount: int
    payment_status: str = "pending"
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    status: str = "confirmed"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BookingCreate(BaseModel):
    event_id: str
    quantity: int

class Payment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    booking_id: str
    amount: int
    currency: str = "INR"
    razorpay_order_id: str
    razorpay_payment_id: Optional[str] = None
    status: str = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PaymentVerify(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    booking_id: str

class Notification(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    message: str
    type: str
    read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ============= AUTH UTILITIES =============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str, email: str) -> str:
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.replace('Bearer ', '')
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({"id": payload['user_id']}, {"_id": 0, "password": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============= MOCK DATA GENERATOR =============

async def seed_mock_events():
    count = await db.events.count_documents({})
    if count > 0:
        return
    
    sports = ["Football", "Basketball", "Tennis", "Badminton", "Yoga", "Cricket", "Swimming", "Running"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai"]
    images = [
        "https://images.unsplash.com/photo-1529900748604-07564a03e7a6?w=800",
        "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=800",
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800",
        "https://images.unsplash.com/photo-1519766304817-4f37bba85e08?w=800",
        "https://images.unsplash.com/photo-1540497077202-7c8a3999166f?w=800",
        "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=800",
        "https://images.unsplash.com/photo-1571902943202-507ec2618e8f?w=800",
        "https://images.unsplash.com/photo-1476480862126-209bfaa8edc8?w=800"
    ]
    
    mock_events = []
    for i in range(20):
        sport = sports[i % len(sports)]
        city = cities[i % len(cities)]
        total_spots = random.randint(10, 30)
        spots_remaining = random.randint(0, total_spots)
        
        event = {
            "id": str(uuid.uuid4()),
            "title": f"{sport} Session - {city}",
            "description": f"Join us for an exciting {sport.lower()} session! All skill levels welcome. Come meet new people and play together.",
            "sport": sport,
            "image": images[i % len(images)],
            "date": (datetime.now(timezone.utc) + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            "time": f"{random.randint(6, 20)}:00",
            "location": f"{city} Sports Complex",
            "city": city,
            "price": random.choice([299, 399, 499, 599, 799]),
            "total_spots": total_spots,
            "spots_remaining": spots_remaining,
            "organizer": "Pause.play Community",
            "venue_details": f"Indoor court at {city} Sports Hub",
            "latitude": 19.0760 + random.uniform(-2, 2),
            "longitude": 72.8777 + random.uniform(-2, 2),
            "is_popular": random.choice([True, False]),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        mock_events.append(event)
    
    await db.events.insert_many(mock_events)
    logger.info(f"Seeded {len(mock_events)} mock events")

# ============= AUTH ROUTES =============

@api_router.post("/auth/register")
async def register(user_input: UserCreate):
    existing = await db.users.find_one({"email": user_input.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=user_input.email,
        name=user_input.name,
        phone=user_input.phone
    )
    
    user_dict = user.model_dump()
    user_dict['password'] = hash_password(user_input.password)
    user_dict['created_at'] = user_dict['created_at'].isoformat()
    
    await db.users.insert_one(user_dict)
    
    token = create_token(user.id, user.email)
    
    return {
        "user": user.model_dump(),
        "token": token
    }

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(credentials.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user['id'], user['email'])
    
    del user['password']
    
    return {
        "user": user,
        "token": token
    }

@api_router.get("/auth/me")
async def get_me(current_user = Depends(get_current_user)):
    return current_user

# ============= EVENTS ROUTES =============

@api_router.get("/events")
async def get_events(
    sport: Optional[str] = None,
    city: Optional[str] = None,
    date: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    search: Optional[str] = None
):
    query = {}
    
    if sport:
        query['sport'] = sport
    if city:
        query['city'] = city
    if date:
        query['date'] = date
    if min_price is not None or max_price is not None:
        query['price'] = {}
        if min_price is not None:
            query['price']['$gte'] = min_price
        if max_price is not None:
            query['price']['$lte'] = max_price
    if search:
        query['$or'] = [
            {'title': {'$regex': search, '$options': 'i'}},
            {'description': {'$regex': search, '$options': 'i'}},
            {'sport': {'$regex': search, '$options': 'i'}}
        ]
    
    events = await db.events.find(query, {"_id": 0}).sort("date", 1).to_list(100)
    return events

@api_router.get("/events/{event_id}")
async def get_event(event_id: str):
    event = await db.events.find_one({"id": event_id}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@api_router.get("/events/trending/list")
async def get_trending_events():
    events = await db.events.find({"is_popular": True}, {"_id": 0}).limit(6).to_list(6)
    if not events:
        events = await db.events.find({}, {"_id": 0}).limit(6).to_list(6)
    return events

@api_router.get("/filters/options")
async def get_filter_options():
    sports = await db.events.distinct("sport")
    cities = await db.events.distinct("city")
    return {
        "sports": sorted(sports),
        "cities": sorted(cities)
    }

# ============= BOOKINGS ROUTES =============

@api_router.post("/bookings")
async def create_booking(booking_input: BookingCreate, current_user = Depends(get_current_user)):
    event = await db.events.find_one({"id": booking_input.event_id}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if event['spots_remaining'] < booking_input.quantity:
        raise HTTPException(status_code=400, detail="Not enough spots available")
    
    total_amount = event['price'] * booking_input.quantity
    
    booking = Booking(
        user_id=current_user['id'],
        event_id=booking_input.event_id,
        quantity=booking_input.quantity,
        total_amount=total_amount
    )
    
    booking_dict = booking.model_dump()
    booking_dict['created_at'] = booking_dict['created_at'].isoformat()
    
    await db.bookings.insert_one(booking_dict)
    
    return booking.model_dump()

@api_router.get("/bookings")
async def get_user_bookings(current_user = Depends(get_current_user)):
    bookings = await db.bookings.find({"user_id": current_user['id']}, {"_id": 0}).sort("created_at", -1).to_list(100)
    
    # Enrich with event details
    for booking in bookings:
        event = await db.events.find_one({"id": booking['event_id']}, {"_id": 0})
        if event:
            booking['event'] = event
    
    return bookings

@api_router.get("/bookings/{booking_id}")
async def get_booking(booking_id: str, current_user = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": booking_id, "user_id": current_user['id']}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    event = await db.events.find_one({"id": booking['event_id']}, {"_id": 0})
    if event:
        booking['event'] = event
    
    return booking

# ============= PAYMENTS ROUTES =============

# @api_router.post("/payments/create-order")
# async def create_payment_order(booking_id: str, current_user = Depends(get_current_user)):
#     booking = await db.bookings.find_one({"id": booking_id, "user_id": current_user['id']}, {"_id": 0})
#     if not booking:
#         raise HTTPException(status_code=404, detail="Booking not found")
    
#     if booking['payment_status'] == 'completed':
#         raise HTTPException(status_code=400, detail="Payment already completed")
    
#     # Create Razorpay order
#     try:
#         order_data = {
#             "amount": booking['total_amount'] * 100,  # Amount in paise
#             "currency": "INR",
#             "receipt": booking_id,
#             "notes": {
#                 "booking_id": booking_id,
#                 "user_id": current_user['id']
#             }
#         }
        
#         # For demo purposes, we'll create a mock order if Razorpay fails
#         try:
#             razorpay_order = razorpay_client.order.create(data=order_data)
#         except:
#             razorpay_order = {
#                 "id": f"order_mock_{str(uuid.uuid4())[:8]}",
#                 "amount": order_data['amount'],
#                 "currency": order_data['currency']
#             }
        
#         # Update booking with order ID
#         await db.bookings.update_one(
#             {"id": booking_id},
#             {"$set": {"razorpay_order_id": razorpay_order['id']}}
#         )
        
#         return {
#             "order_id": razorpay_order['id'],
#             "amount": razorpay_order['amount'],
#             "currency": razorpay_order['currency'],
#             "key_id": RAZORPAY_KEY_ID
#         }
#     except Exception as e:
#         logger.error(f"Error creating payment order: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to create payment order")

@api_router.post("/payments/create-order")
async def create_payment_order(booking_id: str, current_user = Depends(get_current_user)):
    try:
        # Fetch booking
        booking = await db.bookings.find_one(
            {"id": booking_id, "user_id": current_user['id']},
            {"_id": 0}
        )

        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        if booking['payment_status'] == "completed":
            raise HTTPException(status_code=400, detail="Payment already completed")

        # Prepare Razorpay order data
        order_data = {
            "amount": booking['total_amount'] * 100,  # Convert to paise
            "currency": "INR",
            "receipt": booking_id,
            "notes": {
                "booking_id": booking_id,
                "user_id": current_user['id']
            }
        }

        # Create real Razorpay order
        razorpay_order = razorpay_client.order.create(data=order_data)

        # Update booking with real order ID
        await db.bookings.update_one(
            {"id": booking_id},
            {"$set": {"razorpay_order_id": razorpay_order["id"]}}
        )

        return {
            "order_id": razorpay_order["id"],
            "amount": razorpay_order["amount"],
            "currency": razorpay_order["currency"],
            "key_id": RAZORPAY_KEY_ID
        }

    except razorpay.errors.BadRequestError as e:
        logger.error(f"Razorpay Bad Request: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid Razorpay request")

    except razorpay.errors.ServerError as e:
        logger.error(f"Razorpay Server Error: {str(e)}")
        raise HTTPException(status_code=502, detail="Razorpay server error")

    except Exception as e:
        logger.error(f"Payment order creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create payment order")

@api_router.post("/payments/verify")
async def verify_payment(payment_data: PaymentVerify, current_user = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": payment_data.booking_id, "user_id": current_user['id']}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # For demo, we'll skip actual signature verification
    # In production, use: razorpay_client.utility.verify_payment_signature()
    
    # Update booking with payment details
    await db.bookings.update_one(
        {"id": payment_data.booking_id},
        {"$set": {
            "payment_status": "completed",
            "razorpay_payment_id": payment_data.razorpay_payment_id,
            "status": "confirmed"
        }}
    )
    
    # Update event spots
    await db.events.update_one(
        {"id": booking['event_id']},
        {"$inc": {"spots_remaining": -booking['quantity']}}
    )
    
    # Create success notification
    notification = Notification(
        user_id=current_user['id'],
        title="Booking Confirmed!",
        message=f"Your booking for {booking['quantity']} spot(s) has been confirmed.",
        type="success"
    )
    notif_dict = notification.model_dump()
    notif_dict['created_at'] = notif_dict['created_at'].isoformat()
    await db.notifications.insert_one(notif_dict)
    
    return {"status": "success", "message": "Payment verified successfully"}

# ============= NOTIFICATIONS ROUTES =============

@api_router.get("/notifications")
async def get_notifications(current_user = Depends(get_current_user)):
    notifications = await db.notifications.find({"user_id": current_user['id']}, {"_id": 0}).sort("created_at", -1).limit(50).to_list(50)
    return notifications

@api_router.put("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, current_user = Depends(get_current_user)):
    result = await db.notifications.update_one(
        {"id": notification_id, "user_id": current_user['id']},
        {"$set": {"read": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@api_router.get("/notifications/unread/count")
async def get_unread_count(current_user = Depends(get_current_user)):
    count = await db.notifications.count_documents({"user_id": current_user['id'], "read": False})
    return {"count": count}

# ============= ADMIN ROUTES =============

@api_router.get("/admin/stats")
async def get_admin_stats(current_user = Depends(get_current_user)):
    total_events = await db.events.count_documents({})
    total_bookings = await db.bookings.count_documents({})
    total_users = await db.users.count_documents({})
    
    completed_bookings = await db.bookings.count_documents({"payment_status": "completed"})
    
    # Calculate total revenue
    pipeline = [
        {"$match": {"payment_status": "completed"}},
        {"$group": {"_id": None, "total": {"$sum": "$total_amount"}}}
    ]
    revenue_result = await db.bookings.aggregate(pipeline).to_list(1)
    total_revenue = revenue_result[0]['total'] if revenue_result else 0
    
    return {
        "total_events": total_events,
        "total_bookings": total_bookings,
        "total_users": total_users,
        "completed_bookings": completed_bookings,
        "total_revenue": total_revenue
    }

@api_router.get("/admin/bookings")
async def get_all_bookings(current_user = Depends(get_current_user)):
    bookings = await db.bookings.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    
    # Enrich with event and user details
    for booking in bookings:
        event = await db.events.find_one({"id": booking['event_id']}, {"_id": 0})
        user = await db.users.find_one({"id": booking['user_id']}, {"_id": 0, "password": 0})
        if event:
            booking['event'] = event
        if user:
            booking['user'] = user
    
    return bookings

# ============= ROOT ROUTE =============

@api_router.get("/")
async def root():
    return {"message": "Pause.play API", "version": "1.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    await seed_mock_events()
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
