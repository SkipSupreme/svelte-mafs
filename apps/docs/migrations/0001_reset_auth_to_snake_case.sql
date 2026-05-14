-- Reset auth-area tables to match the Drizzle schema.
-- Live D1 inherited camelCase columns (expiresAt, createdAt, userId, ...)
-- from a pre-Drizzle hand migrate; the Better Auth + drizzle adapter now
-- emits snake_case SQL, so every INSERT into user/session/account/
-- verification 500s and login is fully broken. All referencing tables
-- (bookmark, lesson_view, exercise_answer, note, email_drop, user_profile)
-- were empty in prod when this migration ran, so we drop+recreate the
-- whole auth-FK area from migration 0000's canonical body. rate_limit is
-- untouched (no user FK, already snake_case).
DROP TABLE IF EXISTS `account`;
--> statement-breakpoint
DROP TABLE IF EXISTS `session`;
--> statement-breakpoint
DROP TABLE IF EXISTS `user_profile`;
--> statement-breakpoint
DROP TABLE IF EXISTS `bookmark`;
--> statement-breakpoint
DROP TABLE IF EXISTS `lesson_view`;
--> statement-breakpoint
DROP TABLE IF EXISTS `exercise_answer`;
--> statement-breakpoint
DROP TABLE IF EXISTS `note`;
--> statement-breakpoint
DROP TABLE IF EXISTS `email_drop`;
--> statement-breakpoint
DROP TABLE IF EXISTS `verification`;
--> statement-breakpoint
DROP TABLE IF EXISTS `user`;
--> statement-breakpoint
CREATE TABLE `user` (
	`id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`email_verified` integer DEFAULT false NOT NULL,
	`name` text,
	`image` text,
	`role` text DEFAULT 'user' NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `user_email_unique` ON `user` (`email`);
--> statement-breakpoint
CREATE TABLE `session` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`token` text NOT NULL,
	`expires_at` integer NOT NULL,
	`ip_address` text,
	`user_agent` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `session_token_unique` ON `session` (`token`);
--> statement-breakpoint
CREATE TABLE `account` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`provider_id` text NOT NULL,
	`account_id` text NOT NULL,
	`access_token` text,
	`refresh_token` text,
	`id_token` text,
	`access_token_expires_at` integer,
	`refresh_token_expires_at` integer,
	`scope` text,
	`password` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `account_provider_id_account_id_unique` ON `account` (`provider_id`,`account_id`);
--> statement-breakpoint
CREATE TABLE `verification` (
	`id` text PRIMARY KEY NOT NULL,
	`identifier` text NOT NULL,
	`value` text NOT NULL,
	`expires_at` integer NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `user_profile` (
	`user_id` text PRIMARY KEY NOT NULL,
	`display_name` text,
	`marketing_opt_in` integer DEFAULT false NOT NULL,
	`onboarded_at` integer,
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `bookmark` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`lesson_slug` text NOT NULL,
	`anchor` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `bookmark_user_id_lesson_slug_anchor_unique` ON `bookmark` (`user_id`,`lesson_slug`,`anchor`);
--> statement-breakpoint
CREATE TABLE `lesson_view` (
	`user_id` text NOT NULL,
	`course_slug` text NOT NULL,
	`module_slug` text NOT NULL,
	`lesson_slug` text NOT NULL,
	`first_seen_at` integer NOT NULL,
	`last_seen_at` integer NOT NULL,
	`view_count` integer DEFAULT 1 NOT NULL,
	`completed_at` integer,
	PRIMARY KEY(`user_id`, `lesson_slug`),
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `lesson_view_by_course_recency` ON `lesson_view` (`user_id`,`course_slug`,`last_seen_at`);
--> statement-breakpoint
CREATE TABLE `exercise_answer` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`lesson_slug` text NOT NULL,
	`exercise_id` text NOT NULL,
	`answer_json` text NOT NULL,
	`is_correct` integer,
	`attempt_no` integer DEFAULT 1 NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `exercise_answer_by_user_lesson` ON `exercise_answer` (`user_id`,`lesson_slug`);
--> statement-breakpoint
CREATE TABLE `note` (
	`user_id` text NOT NULL,
	`lesson_slug` text NOT NULL,
	`body` text DEFAULT '' NOT NULL,
	`updated_at` integer NOT NULL,
	PRIMARY KEY(`user_id`, `lesson_slug`),
	FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `email_drop` (
	`id` text PRIMARY KEY NOT NULL,
	`subject` text NOT NULL,
	`body_md` text NOT NULL,
	`course_slug` text,
	`module_slug` text,
	`lesson_slug` text,
	`target_count` integer NOT NULL,
	`sent_count` integer NOT NULL,
	`sent_at` integer NOT NULL,
	`sent_by_user_id` text,
	FOREIGN KEY (`sent_by_user_id`) REFERENCES `user`(`id`) ON UPDATE no action ON DELETE no action
);
