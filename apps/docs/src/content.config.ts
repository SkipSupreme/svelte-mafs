import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

/**
 * Content hierarchy: Course → Arc → Module → Lesson.
 *
 * - A Course is the whole product ("Machine Learning, Backpropagation and
 *   AI — The Math"). The course file owns the arc list.
 * - A Module is a unit of ~1–5 lessons teaching one cluster of concepts
 *   (e.g. "Single-variable calculus — derivatives & chain rule").
 * - A Lesson is one read-in-one-sitting interactive MDX page.
 *
 * Arcs are declared inline on the Course rather than as their own
 * collection — they're structural grouping, not authored content.
 */

const courses = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/courses' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    summary: z.string(),
    hero: z.string().optional(),
    heroEndpoint: z.string().optional(),
    status: z.enum(['alpha', 'beta', 'shipped']).default('alpha'),
    arcs: z
      .array(
        z.object({
          id: z.string(),
          title: z.string(),
          description: z.string(),
          modules: z.array(z.string()),
        }),
      ),
    draft: z.boolean().default(false),
  }),
});

const modules = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/modules' }),
  schema: z.object({
    course: z.string(),
    arc: z.string(),
    order: z.number().int(),
    title: z.string(),
    summary: z.string(),
    status: z.enum(['planned', 'drafting', 'shipped']).default('planned'),
    estimatedMinutes: z.number().int().positive().optional(),
    prereqs: z.array(z.string()).default([]),
    conceptsCovered: z.array(z.string()).default([]),
    /** The "where this shows up in the transformer" callback shown at the
       end of every lesson in this module. */
    endgameConnection: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

const lessons = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/lessons' }),
  schema: z.object({
    title: z.string(),
    /** Module id, e.g. `m5-calculus`. References modules collection. */
    module: z.string(),
    summary: z.string(),
    estimatedMinutes: z.number().int().positive(),
    /** Ordering within the module (1-based). */
    order: z.number().int().default(0),
    prereqs: z.array(z.string()).default([]),
    conceptsCovered: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

export const collections = { courses, modules, lessons };
