from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
 
from app.core.database import init_db
from app.core.ml_engine import MLEngine
from app.api import wells, predictions, dashboard, health
