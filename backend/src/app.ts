import compression from 'compression';
import cookieParser from 'cookie-parser';
import cors from 'cors';
import express, { type Application } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import passport from 'passport';
import config from './config/index';
import './config/passport.js';
import { globalErrorHandler } from './middlewares/global-error-handler.js';
import { notFound } from './middlewares/not-found.js';
import { apiRouter } from './routes/index.js';

const app: Application = express();

app.set('trust proxy', 1);
app.disable('x-powered-by');

const rawOrigin = config.frontend_url || 'http://localhost:3000';
const allowedOrigins = new Set([
  rawOrigin,
  'http://localhost:3000',
  'http://localhost:3010',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:3010',
]);

if (rawOrigin.includes('.sslip.io')) {
  allowedOrigins.add(rawOrigin.replace('.sslip.io', ''));
}
const ipMatch = rawOrigin.match(/https?:\/\/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?/);
if (ipMatch) {
  const ip = ipMatch[1];
  const port = ipMatch[2] || '';
  allowedOrigins.add(`http://${ip}.sslip.io${port}`);
  allowedOrigins.add(`https://${ip}.sslip.io${port}`);
}

// Security & utility middlewares
app.use(
  helmet({
    crossOriginResourcePolicy: { policy: 'cross-origin' },
  }),
);
app.use(
  cors({
    credentials: true,
    origin: (origin, callback) => {
      if (!origin || allowedOrigins.has(origin) || config.node_env !== 'production') {
        return callback(null, true);
      }
      return callback(new Error('Not allowed by CORS'), false);
    },
  }),
);
app.use(compression());
app.use(cookieParser());
app.use(express.json({ limit: '8mb' }));
app.use(express.urlencoded({ extended: true, limit: '1mb' }));
app.use(passport.initialize());

// Rate limiter
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    limit: 100,
    standardHeaders: 'draft-8',
    legacyHeaders: false,
  }),
);

// Root route
app.get('/', (_req, res) => res.json({ success: true, message: 'Server is running' }));

// Application routes
app.use('/api/v1', apiRouter);

// Error handling middlewares
app.use(notFound);
app.use(globalErrorHandler);

export default app;
