import { GlassCard } from "@/components/ui/glass";
import { Check, X } from "lucide-react";
import Link from "next/link";

const tiers = [
    {
        name: "Free",
        price: "$0",
        description: "Basic access to odds and scores.",
        features: [
            "Live Scores & Odds",
            "Basic Team Stats",
            "Public Betting Percentages",
            "Limited Historical Data",
        ],
        notIncluded: [
            "Model Predictions",
            "Edge Analysis",
            "Line Movement Charts",
            "Confidence Pool",
        ],
        cta: "Get Started",
        href: "/games",
        featured: false,
    },
    {
        name: "Pro",
        price: "$29",
        period: "/mo",
        description: "Full access to our predictive models.",
        features: [
            "Everything in Free",
            "Unlimited Model Predictions",
            "Real-time Edge Alerts",
            "Advanced Analytics Charts",
            "Full Historical Database",
            "Priority Support",
        ],
        notIncluded: [],
        cta: "Start Free Trial",
        href: "/signup",
        featured: true,
    },
];

export default function PricingPage() {
    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div className="text-center mb-16">
                <h1 className="text-4xl font-black text-white mb-4">
                    Unlock the <span className="text-emerald-400">Edge</span>
                </h1>
                <p className="text-xl text-slate-400 max-w-2xl mx-auto">
                    Stop guessing. Start winning with our advanced predictive models and real-time market analysis.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
                {tiers.map((tier) => (
                    <GlassCard
                        key={tier.name}
                        className={`relative flex flex-col p-8 ${tier.featured ? 'border-emerald-500/50 shadow-[0_0_30px_rgba(16,185,129,0.1)]' : ''}`}
                    >
                        {tier.featured && (
                            <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 px-4 py-1 bg-emerald-500 rounded-full text-xs font-bold text-slate-950 uppercase tracking-wider">
                                Most Popular
                            </div>
                        )}

                        <div className="mb-8">
                            <h3 className="text-xl font-bold text-white mb-2">{tier.name}</h3>
                            <div className="flex items-baseline gap-1">
                                <span className="text-4xl font-black text-white">{tier.price}</span>
                                {tier.period && <span className="text-slate-400">{tier.period}</span>}
                            </div>
                            <p className="text-slate-400 mt-2 text-sm">{tier.description}</p>
                        </div>

                        <div className="flex-1 space-y-4 mb-8">
                            {tier.features.map((feature) => (
                                <div key={feature} className="flex items-start gap-3">
                                    <Check className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                                    <span className="text-slate-300 text-sm">{feature}</span>
                                </div>
                            ))}
                            {tier.notIncluded.map((feature) => (
                                <div key={feature} className="flex items-start gap-3 opacity-50">
                                    <X className="w-5 h-5 text-slate-500 flex-shrink-0" />
                                    <span className="text-slate-500 text-sm">{feature}</span>
                                </div>
                            ))}
                        </div>

                        <Link
                            href={tier.href}
                            className={`w-full py-3 rounded-lg font-bold text-center transition-all duration-200 ${tier.featured
                                    ? 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 hover:shadow-[0_0_20px_rgba(52,211,153,0.4)]'
                                    : 'bg-slate-800 text-white hover:bg-slate-700'
                                }`}
                        >
                            {tier.cta}
                        </Link>
                    </GlassCard>
                ))}
            </div>
        </div>
    );
}
