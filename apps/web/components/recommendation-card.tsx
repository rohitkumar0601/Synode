import { ShieldAlert, Trophy } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Recommendation } from "@/lib/types";

export function RecommendationCard({ rec }: { rec: Recommendation }) {
  const notes = rec.ranking.flatMap((c) => (c.notes ?? []).map((n) => ({ generator: c.generator, note: n })));
  return (
    <section aria-label="Recommendation">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-muted-foreground">Recommendation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {rec.winner ? (
            <p className="flex items-center gap-2 text-2xl font-semibold">
              <Trophy aria-hidden="true" className="size-6 text-primary" />
              <span>{rec.winner}</span>
            </p>
          ) : (
            <p className="flex items-center gap-2 text-xl font-semibold">
              <ShieldAlert aria-hidden="true" className="size-6 text-destructive" />
              <span>No generator passed the privacy gate</span>
            </p>
          )}
          <p className="text-sm">{rec.reason}</p>
          {notes.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {notes.map((n) => (
                <li key={`${n.generator}:${n.note}`}>
                  <span className="font-medium text-foreground">{n.generator}:</span> {n.note}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
